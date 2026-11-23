function generate_vienna_channels(simulatorPath, outDir, nRealizations, seed, modelNames, fileSuffix)
%GENERATE_VIENNA_CHANNELS  Ground-truth 72x14 channel grids with the Vienna 5G Link-Level Simulator.
%
%   generate_vienna_channels(simulatorPath, outDir, nRealizations, seed)
%
%   simulatorPath : folder of the extracted 5GLLSimulator (contains +Channel/FastFading.m)
%   outDir        : output folder, e.g. '../data'  -> vienna_VehA.mat, vienna_SUI5.mat
%   nRealizations : number of independent sub-frames per channel model (paper: 40000)
%   seed          : RNG seed (reproducibility)
%   modelNames    : (optional) subset, e.g. {'SUI5'}; default {'VehA','SUI5'}
%   fileSuffix    : (optional) appended to the file name, e.g. '_part1', to generate one model in
%                   several chunks in parallel MATLAB sessions (use a different seed per chunk).
%                   The Python loader concatenates all files listed in channel.mat_files.
%
%   Set-up (Soltani et al. 2019, Sec. IV): SISO, LTE-like grid with 72 sub-carriers x 14 OFDM
%   symbols (15 kHz spacing, normal CP, 1 ms sub-frame), carrier 2.1 GHz, UE speed 50 km/h,
%   channel models Vehicular-A and SUI5.
%
%   The channel is produced by Channel.FastFading of the Vienna simulator (time-variant
%   tapped-delay line, Jakes Doppler spectrum, WSSUS sum-of-paths model). The perfect channel
%   transfer function of every OFDM symbol is read with FastFading.GetTransferFunction at the
%   centre of the useful part of the symbol, on the 72 active sub-carriers (k = -36..-1, 1..36).
%
%   Sampling rate 15.36 MHz (N_FFT = 1024) so that the VehA tap delays (multiples of 10 ns in the
%   ITU table) are represented with 65 ns resolution. SUI5 is not built into the simulator and is
%   passed as a self-defined power-delay profile (0 / 4 / 10 us, 0 / -5 / -10 dB).
%
%   Output: variable H, single complex [N x 72 x 14] (normalised to unit power by the Python loader),
%   and struct meta with all settings.

if nargin < 1 || isempty(simulatorPath), simulatorPath = fullfile('..', '..', 'supporting', '5GLLSimulator'); end
if nargin < 2 || isempty(outDir),        outDir = fullfile('..', 'data'); end
if nargin < 3 || isempty(nRealizations), nRealizations = 40000; end
if nargin < 4 || isempty(seed),          seed = 2026; end
if nargin < 5 || isempty(modelNames),    modelNames = {'VehA', 'SUI5'}; end
if nargin < 6,                           fileSuffix = ''; end
if ischar(modelNames), modelNames = {modelNames}; end

addpath(simulatorPath);
if ~exist(outDir, 'dir'), mkdir(outDir); end
rng(seed, 'twister');

% ------------------------------------------------------------------ system parameters
SubcarrierSpacing = 15e3;
FFTSize           = 1024;                          % fs = 15.36 MHz
SamplingRate      = SubcarrierSpacing * FFTSize;
CP                = [80 72 72 72 72 72 72 80 72 72 72 72 72 72];   % normal CP at 15.36 MHz
NSymbols          = 14;
NSubcarriers      = 72;
CarrierFrequency  = 2.1e9;
Velocity_kmh      = 50;
MaxDoppler        = Velocity_kmh/3.6 * CarrierFrequency / 2.998e8;   % ~97.2 Hz
SamplesTotal      = sum(CP) + NSymbols*FFTSize;    % 15360 samples = 1 ms
NPaths            = 50;                            % sum-of-paths terms per tap (Jakes)
SubSampling       = 32;                            % time sub-sampling of the fading process (0.5 % of T_coh)

% time instant (1-based sample index) at the centre of the useful part of each OFDM symbol
symStart = [0 cumsum(CP(1:end-1) + FFTSize)];
TimePos  = symStart + CP + FFTSize/2 + 1;

% active sub-carriers: logical k = -36..-1, +1..+36  -> FFT bins (1-based)
k = [-NSubcarriers/2:-1, 1:NSubcarriers/2];
ActiveBins = mod(k, FFTSize) + 1;

% ------------------------------------------------------------------ channel models
dt = 1/SamplingRate;
sui5 = zeros(1, round(10e-6/dt) + 1);            % self-defined PDP on the sampling grid
sui5(round(0/dt) + 1)    = 10^(0/10);
sui5(round(4e-6/dt) + 1) = 10^(-5/10);
sui5(round(10e-6/dt) + 1) = 10^(-10/10);

models = {'VehA', 'VehicularA'; 'SUI5', sui5};
models = models(ismember(models(:, 1), modelNames), :);
dummy.spatialChannelModel = false;

for m = 1:size(models, 1)
    name = models{m, 1};
    fprintf('=== %s: %d realisations, f_D = %.1f Hz, fs = %.2f MHz ===\n', name, nRealizations, MaxDoppler, SamplingRate/1e6);
    ch = Channel.FastFading( ...
        SamplingRate, ...          % sampling rate
        models{m, 2}, ...          % power delay profile
        SamplesTotal, ...          % samples per realisation (one sub-frame)
        MaxDoppler, ...            % maximum Doppler shift
        'Jakes', ...               % Doppler spectrum
        NPaths, ...                % paths of the WSSUS process
        false, ...                 % independent realisations (no correlation across frames)
        'none', 0, 0, ...          % no spatial correlation (SISO)
        1, 1, ...                  % 1 Tx, 1 Rx
        false, ...                 % no warning about delay rounding
        0, 1, ...                  % TWDP K = 0 (Rayleigh), delta
        dummy, [1 1], [1 1], ...   % no 3-D spatial model
        CarrierFrequency, ...
        SubSampling);
    fprintf('RMS delay spread: %.3f us\n', ch.GetRmsDelaySpread*1e6);

    H = complex(zeros(nRealizations, NSubcarriers, NSymbols, 'single'));
    tic;
    for n = 1:nRealizations
        ch.NewRealization(n);
        Hn = ch.GetTransferFunction(TimePos, FFTSize, ActiveBins);   % [72 x 14]
        H(n, :, :) = single(Hn);
        if mod(n, 5000) == 0
            fprintf('  %d / %d  (%.0f s)\n', n, nRealizations, toc);
        end
    end
    p = mean(abs(H(:)).^2);
    fprintf('mean |H|^2 = %.4f (normalised to 1 by the Python loader)\n', p);
    meta = struct('model', name, 'samplingRate', SamplingRate, 'fftSize', FFTSize, ...
        'subcarrierSpacing', SubcarrierSpacing, 'carrierFrequency', CarrierFrequency, ...
        'velocity_kmh', Velocity_kmh, 'maxDoppler', MaxDoppler, 'nPaths', NPaths, ...
        'seed', seed, 'timePos', TimePos, 'activeBins', ActiveBins, ...
        'rmsDelaySpread', ch.GetRmsDelaySpread, 'simulator', 'Vienna 5G LLS Channel.FastFading');
    f = fullfile(outDir, ['vienna_' name fileSuffix '.mat']);
    save(f, 'H', 'meta', '-v7');
    fprintf('saved %s  [%d x %d x %d]\n', f, size(H, 1), size(H, 2), size(H, 3));
end
end
