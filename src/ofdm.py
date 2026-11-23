"""SISO-OFDM system (LTE-like 1.4 MHz numerology) and pilot handling.

Resource grid
-------------
    K = 72 sub-carriers (6 RBs x 12),  L = 14 OFDM symbols (1 ms sub-frame),
    delta_f = 15 kHz,  N_FFT = 128  ->  f_s = N_FFT * delta_f = 1.92 MHz,
    normal CP = [10, 9, 9, 9, 9, 9, 9] samples per slot (~5.2 / 4.7 us).
The 72 used sub-carriers are k = -36..-1, +1..+36 around the (unused) DC bin.

Received signal model on the grid (CP longer than the delay spread, channel constant
over one OFDM symbol):
    Y[k, l] = H[k, l] X[k, l] + Z[k, l],     Z ~ CN(0, sigma^2)
With unit-power pilots and E|H|^2 = 1 the SNR is  SNR = 1 / sigma^2.

LS estimate at a pilot position p:
    H_LS[p] = Y[p] / X[p] = H[p] + Z[p] / X[p]          (unbiased, error variance sigma^2)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ============================================================================ grid
@dataclass
class OFDMGrid:
    n_subcarriers: int = 72
    n_symbols: int = 14
    subcarrier_spacing: float = 15e3
    fft_size: int = 128
    cp_lengths: list = field(default_factory=lambda: [10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9])

    @classmethod
    def from_config(cls, cfg: dict) -> "OFDMGrid":
        o = cfg["ofdm"]
        return cls(int(o["n_subcarriers"]), int(o["n_symbols"]), float(o["subcarrier_spacing"]),
                   int(o["fft_size"]), list(o["cp_lengths"]))

    @property
    def shape(self):
        return (self.n_subcarriers, self.n_symbols)

    @property
    def fs(self) -> float:
        """Sampling rate (Hz)."""
        return self.fft_size * self.subcarrier_spacing

    @property
    def subcarrier_indices(self) -> np.ndarray:
        """Logical sub-carrier indices k = -K/2..-1, 1..K/2 (DC excluded, as in LTE downlink)."""
        h = self.n_subcarriers // 2
        return np.concatenate([np.arange(-h, 0), np.arange(1, h + 1)])

    @property
    def subcarrier_freqs(self) -> np.ndarray:
        """Base-band frequency (Hz) of each used sub-carrier."""
        return self.subcarrier_indices * self.subcarrier_spacing

    @property
    def fft_bins(self) -> np.ndarray:
        """FFT bin of each used sub-carrier."""
        return np.mod(self.subcarrier_indices, self.fft_size)

    @property
    def symbol_starts(self) -> np.ndarray:
        """First sample (start of CP) of every OFDM symbol inside the sub-frame."""
        lens = self.fft_size + np.asarray(self.cp_lengths)
        return np.concatenate([[0], np.cumsum(lens)[:-1]])

    @property
    def symbol_times(self) -> np.ndarray:
        """Time instant (s) at the centre of the useful part of each OFDM symbol.

        The fading process is sampled at these instants (block fading within one symbol)."""
        return (self.symbol_starts + np.asarray(self.cp_lengths) + self.fft_size / 2) / self.fs

    @property
    def frame_length(self) -> int:
        return int(self.n_symbols * self.fft_size + sum(self.cp_lengths))


# ============================================================================ pilots
def lattice_pilot_mask(grid: OFDMGrid, symbols, step, offsets) -> np.ndarray:
    """Lattice (diamond) pilot pattern.

    mask[k, l] = True  if  l == symbols[i]  and  k = offsets[i] + m * step,  m = 0, 1, ...
    Staggering the offsets between pilot symbols gives the diamond-shaped lattice used by LTE.
    """
    K, L = grid.shape
    mask = np.zeros((K, L), dtype=bool)
    for l, off in zip(symbols, offsets):
        mask[np.arange(int(off), K, int(step)), int(l)] = True
    return mask


def pilot_mask(cfg: dict, grid: OFDMGrid, n_pilots: int) -> np.ndarray:
    pat = cfg["pilots"]["patterns"][int(n_pilots)]
    mask = lattice_pilot_mask(grid, pat["symbols"], pat["step"], pat["offsets"])
    if mask.sum() != int(n_pilots):
        raise ValueError(f"pilot pattern for {n_pilots} pilots produced {mask.sum()} pilots")
    return mask


def pilot_flat_indices(mask: np.ndarray) -> np.ndarray:
    """Row-major flat indices (k * L + l) of the pilot positions (same order as np.nonzero)."""
    return np.flatnonzero(mask.ravel())


# ============================================================================ modulation
def qpsk(shape, rng: np.random.Generator) -> np.ndarray:
    """Unit-energy QPSK symbols (used for pilots and data)."""
    b = rng.integers(0, 2, size=(2,) + tuple(shape))
    return ((2 * b[0] - 1) + 1j * (2 * b[1] - 1)) / np.sqrt(2)


def ofdm_modulate(grid: OFDMGrid, X: np.ndarray) -> np.ndarray:
    """Resource grid X[..., K, L] -> time-domain sub-frame x[..., N_frame] (unitary IFFT + CP)."""
    lead = X.shape[:-2]
    F = np.zeros(lead + (grid.fft_size, grid.n_symbols), dtype=complex)
    F[..., grid.fft_bins, :] = X
    x = np.fft.ifft(F, axis=-2) * np.sqrt(grid.fft_size)
    parts = []
    for l, cp in enumerate(grid.cp_lengths):
        s = x[..., :, l]
        parts.append(np.concatenate([s[..., -cp:], s], axis=-1))
    return np.concatenate(parts, axis=-1)


def ofdm_demodulate(grid: OFDMGrid, y: np.ndarray) -> np.ndarray:
    """Time-domain sub-frame y[..., N_frame] -> resource grid Y[..., K, L] (CP removal + FFT)."""
    out = []
    for l, (st, cp) in enumerate(zip(grid.symbol_starts, grid.cp_lengths)):
        seg = y[..., st + cp: st + cp + grid.fft_size]
        Yf = np.fft.fft(seg, axis=-1) / np.sqrt(grid.fft_size)
        out.append(Yf[..., grid.fft_bins])
    return np.stack(out, axis=-1)


def snr_to_noise_var(snr_db) -> np.ndarray:
    """sigma^2 = 10^(-SNR/10)  (unit signal power)."""
    return 10.0 ** (-np.asarray(snr_db, dtype=float) / 10.0)


def ls_pilot_observation(H: np.ndarray, mask: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    """Transmit QPSK pilots through the channel and return the LS estimates at the pilots.

    Parameters
    ----------
    H      : (N, K, L) complex ground-truth channel grids
    mask   : (K, L) boolean pilot mask
    snr_db : scalar or (N,) array (per-frame SNR)
    Returns
    -------
    H_LS   : (N, P) complex64,  H_LS = Y_p / X_p = H_p + Z_p / X_p
    """
    N = H.shape[0]
    idx = pilot_flat_indices(mask)
    Hp = H.reshape(N, -1)[:, idx]
    Xp = qpsk(Hp.shape, rng)                                     # known pilot symbols
    sigma2 = np.broadcast_to(snr_to_noise_var(snr_db), (N,))[:, None]
    Z = np.sqrt(sigma2 / 2) * (rng.standard_normal(Hp.shape) + 1j * rng.standard_normal(Hp.shape))
    Yp = Hp * Xp + Z                                              # received pilots
    return (Yp / Xp).astype(np.complex64)                         # LS estimate


def simulate_time_domain_link(grid: OFDMGrid, taps: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Full time-domain OFDM link over a *static, sample-spaced* channel (used for verification).

    taps : (n_taps,) complex impulse response (n_taps - 1 <= min CP length)
    X    : (K, L) transmitted grid.     Returns the demodulated grid Y (K, L).
    """
    x = ofdm_modulate(grid, X)
    y = np.convolve(x, taps)[: x.shape[-1]]
    return ofdm_demodulate(grid, y)
