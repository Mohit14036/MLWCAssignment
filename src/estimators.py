"""Conventional pilot-aided channel estimators.

All estimators are *linear* in the LS pilot vector h_p (P x 1) because the pilot positions are
fixed, so each one is a (KL x P) matrix W and the full-grid estimate is  h_hat = W h_p.

1. LS + 2-D interpolation  (W = interpolation operator)
       'linear'  : Delaunay-triangulation piece-wise linear (scipy griddata), nearest outside hull
       'cubic'   : Clough-Tocher piece-wise cubic, nearest outside hull
       'rbf'     : Gaussian radial-basis-function interpolation (as in the ChannelNet reference code),
                   phi(r) = exp(-(r/eps)^2), eps = mean pilot spacing (scipy.interpolate.Rbf default)

2. LMMSE (Wiener) estimation           h_hat = R_hp (R_pp + sigma^2 I)^{-1} h_LS                (3)
       R_hp = E[h h_p^H]  (KL x P),  R_pp = E[h_p h_p^H]  (P x P)
   * Ideal MMSE     : R = exact 2-D channel covariance (sample covariance of the 32k training channels)
   * Estimated MMSE : R estimated from a finite set of noisy LS+linear-interpolated estimates,
                      R_hat = mean(h_int h_int^H) - sigma^2 W_lin W_lin^T   (noise bias removed)
   * Ideal ALMMSE   : approximate LMMSE of Simko et al. (WSA 2010), the third baseline of the paper:
                      Kronecker model R ~ R_time (x) R_freq with ideal statistics, separate rank-one
                      eigen-filtering of the linearly interpolated LS grid in frequency and time
   * Robust LMMSE   : Li, Cimini & Sollenberger (1998) - R = R_f (x) R_t with a uniform PDP over
                      [0, tau_max] and a Jakes time correlation J0(2 pi f_D dt); needs no statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator, griddata
from scipy.special import j0

from .channel_models import max_doppler_hz
from .ofdm import OFDMGrid, pilot_flat_indices, snr_to_noise_var


# ============================================================================ interpolation
def _grid_points(K, L):
    gk, gl = np.meshgrid(np.arange(K), np.arange(L), indexing="ij")
    return np.column_stack([gk.ravel(), gl.ravel()]).astype(float)        # row-major (k, l)


def rbf_default_epsilon(points: np.ndarray) -> float:
    """scipy.interpolate.Rbf default: average distance between nodes (bounding-box based)."""
    edges = points.max(axis=0) - points.min(axis=0)
    edges = edges[edges > 0]
    return float(np.power(np.prod(edges) / points.shape[0], 1.0 / edges.size))


def interpolation_matrix(mask: np.ndarray, method: str) -> np.ndarray:
    """(K*L, P) real matrix W such that  vec(H_interp) = W @ h_p  (row-major vec).

    Built by interpolating the P unit vectors (interpolation is linear in the data)."""
    K, L = mask.shape
    pk, pl = np.nonzero(mask)                         # same order as pilot_flat_indices
    pts = np.column_stack([pk, pl]).astype(float)
    grid = _grid_points(K, L)
    eye = np.eye(len(pk))
    if method == "nearest":
        W = NearestNDInterpolator(pts, eye)(grid)
    elif method in ("linear", "cubic"):
        W = griddata(pts, eye, grid, method=method)
        bad = np.isnan(W).any(axis=1)                 # outside the convex hull of the pilots
        if bad.any():
            W[bad] = NearestNDInterpolator(pts, eye)(grid[bad])
    elif method == "rbf":
        eps = rbf_default_epsilon(pts)
        # RBFInterpolator's gaussian kernel is exp(-(epsilon*r)^2)  ->  epsilon = 1/eps
        W = RBFInterpolator(pts, eye, kernel="gaussian", epsilon=1.0 / eps)(grid)
    else:
        raise ValueError(f"unknown interpolation method {method}")
    return np.asarray(W, dtype=np.float64)


def apply_linear(W: np.ndarray, h_p: np.ndarray, grid_shape) -> np.ndarray:
    """h_p (N, P) -> estimated grids (N, K, L)."""
    return (h_p @ W.T).reshape((h_p.shape[0],) + tuple(grid_shape)).astype(np.complex64)


# ============================================================================ LMMSE
def sample_covariance(H: np.ndarray) -> np.ndarray:
    """R = E[vec(H) vec(H)^H] estimated from (N, K, L) samples (zero-mean channel)."""
    X = H.reshape(H.shape[0], -1).astype(np.complex128)
    return (X.T @ X.conj()) / X.shape[0]


def robust_covariance(grid: OFDMGrid, tau_max: float, fd_max: float) -> np.ndarray:
    """Separable 2-D covariance of the robust LMMSE (Li et al. 1998).

    r_f(df) = (1 - exp(-j 2 pi df tau_max)) / (j 2 pi df tau_max)    (uniform PDP on [0, tau_max])
    r_t(dt) = J0(2 pi f_D dt)                                          (Jakes spectrum)
    R = R_f kron R_t   (matches row-major vec with index k*L + l)
    """
    f = grid.subcarrier_freqs
    df = f[:, None] - f[None, :]
    x = 2 * np.pi * df * tau_max
    with np.errstate(divide="ignore", invalid="ignore"):
        rf = np.where(df == 0, 1.0 + 0j, (1 - np.exp(-1j * x)) / (1j * x))
    t = grid.symbol_times
    rt = j0(2 * np.pi * fd_max * (t[:, None] - t[None, :]))
    return np.kron(rf, rt)


def lmmse_matrix(R: np.ndarray, pilot_idx: np.ndarray, noise_var: float, loading: float = 1e-8) -> np.ndarray:
    """W = R_hp (R_pp + sigma^2 I)^{-1}   (eq. 3), shape (KL, P)."""
    Rhp = R[:, pilot_idx]
    Rpp = R[np.ix_(pilot_idx, pilot_idx)]
    A = Rpp + (noise_var + loading * np.real(np.trace(Rpp)) / len(pilot_idx)) * np.eye(len(pilot_idx))
    # W A = Rhp  ->  A^T W^T = Rhp^T
    return np.linalg.solve(A.T, Rhp.T).T


def time_freq_covariances(H: np.ndarray):
    """Separate frequency (K x K) and time (L x L) correlation matrices of the channel, so that
    R_h ~ R_time (x) R_freq  (Kronecker / independent time-frequency correlation, eq. (7) of Simko)."""
    Hc = H.astype(np.complex128)
    N, K, L = Hc.shape
    R_f = np.einsum("nkl,njl->kj", Hc, Hc.conj()) / (N * L)
    R_t = np.einsum("nkl,nkm->lm", Hc, Hc.conj()) / (N * K)
    return R_t, R_f


def almmse_matrix(R_t: np.ndarray, R_f: np.ndarray, W_int: np.ndarray, noise_var: float,
                  n_time: int = 2, n_freq: int = 5) -> np.ndarray:
    """ALMMSE of Simko, Mehlfuehrer, Wrulich & Rupp (WSA 2010), the "ideal ALMMSE" baseline of [1].

    The LMMSE filter is approximated by separate filtering of the *interpolated* LS grid in
    frequency and time:   H_hat = B_freq H_LS,int C_time^T                              (17)
      R_freq = U_f D_f U_f^H ,  R_time = U_t D_t U_t^H   (truncated to n_freq / n_time eigenvalues)
      M = (lam_t lam_f^T) ./ (lam_t lam_f^T + sigma^2)                                      (14)
      rank-one approximation  M ~ s_max u v^T  ->  lam_C = s_max u,  lam_B = v               (15,16)
      B_freq = U_f diag(lam_B) U_f^H ,  C_time = U_t diag(lam_C) U_t^H                      (18,19)
    With row-major vec, vec(B X C^T) = (B kron C) vec(X), so the overall linear map is
    (B kron C) W_int, shape (KL, P).  Paper settings: N_freq = 5, N_time = 2.
    """
    lt, Ut = np.linalg.eigh(R_t)
    lf, Uf = np.linalg.eigh(R_f)
    ot, of = np.argsort(lt)[::-1][:n_time], np.argsort(lf)[::-1][:n_freq]
    lt, Ut, lf, Uf = np.clip(lt[ot], 0, None), Ut[:, ot], np.clip(lf[of], 0, None), Uf[:, of]
    P = np.outer(lt, lf)
    M = P / (P + noise_var)
    U, S, Vh = np.linalg.svd(M)
    lam_C, lam_B = S[0] * U[:, 0], Vh[0, :]
    B = (Uf * lam_B) @ Uf.conj().T
    C = (Ut * lam_C) @ Ut.conj().T
    return np.kron(B, C) @ W_int


# ============================================================================ bank
class ConventionalEstimators:
    """All conventional estimators for one (channel model, pilot pattern)."""

    def __init__(self, cfg: dict, mask: np.ndarray, H_train: np.ndarray, rng_factory):
        self.cfg = cfg
        self.grid = OFDMGrid.from_config(cfg)
        self.mask = mask
        self.pidx = pilot_flat_indices(mask)
        self.shape = self.grid.shape
        self.interp = {m: interpolation_matrix(mask, m) for m in cfg["estimators"]["ls_interps"]}
        if "linear" not in self.interp:
            self.interp["linear"] = interpolation_matrix(mask, "linear")
        self.R_ideal = sample_covariance(H_train)
        self.R_t, self.R_f = time_freq_covariances(H_train)
        fd = max_doppler_hz(float(cfg["channel"]["speed_kmh"]), float(cfg["channel"]["carrier_freq"]))
        self.R_robust = robust_covariance(self.grid, float(cfg["estimators"]["robust_tau_max"]), fd)
        n_est = int(cfg["dataset"].get("n_est_cov", 2000))
        self.H_est = H_train[:n_est]
        self.rng_factory = rng_factory          # callable(snr) -> np.random.Generator

    def _estimated_cov(self, snr_db: float) -> np.ndarray:
        """Covariance estimated from noisy LS + linear-interpolation estimates (no genie knowledge)."""
        from .ofdm import ls_pilot_observation
        hp = ls_pilot_observation(self.H_est, self.mask, snr_db, self.rng_factory(snr_db))
        W = self.interp["linear"]
        Hi = apply_linear(W, hp, self.shape)
        C = sample_covariance(Hi)
        return C - snr_to_noise_var(snr_db) * (W @ W.T)

    def estimate_all(self, h_ls: np.ndarray, snr_db: float) -> dict:
        """Return {method name: (N, K, L) estimate} for the LS pilot matrix h_ls (N, P)."""
        s2 = float(snr_to_noise_var(snr_db))
        out = {}
        for m, W in self.interp.items():
            if m in self.cfg["estimators"]["ls_interps"]:
                out[f"LS-{m}"] = apply_linear(W, h_ls, self.shape)
        out["MMSE-robust"] = apply_linear(lmmse_matrix(self.R_robust, self.pidx, s2), h_ls, self.shape)
        out["MMSE-estimated"] = apply_linear(lmmse_matrix(self._estimated_cov(snr_db), self.pidx, s2),
                                             h_ls, self.shape)
        a = self.cfg["estimators"].get("almmse", {"n_time": 2, "n_freq": 5})
        out["ALMMSE-ideal"] = apply_linear(almmse_matrix(self.R_t, self.R_f, self.interp["linear"], s2,
                                                         int(a["n_time"]), int(a["n_freq"])), h_ls, self.shape)
        out["MMSE-ideal"] = apply_linear(lmmse_matrix(self.R_ideal, self.pidx, s2), h_ls, self.shape)
        return out


def mse(H_hat: np.ndarray, H: np.ndarray) -> float:
    """MSE = E |H_hat - H|^2 averaged over all resource elements and frames."""
    return float(np.mean(np.abs(H_hat.astype(np.complex128) - H) ** 2))
