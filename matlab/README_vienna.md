# Channel generation with the Vienna 5G Link-Level Simulator

`generate_vienna_channels.m` produces the ground-truth channel data sets of the assignment with the
course-provided Vienna 5G Link-Level Simulator (`supporting/5GLLSimulator`). It uses the simulator's
time-variant fading class `Channel.FastFading`.

| setting | value |
|---|---|
| sampling rate / FFT | 15.36 MHz / 1024 (15 kHz sub-carriers) |
| grid | 72 active sub-carriers (k = -36..-1, 1..36) x 14 OFDM symbols, normal CP (1 ms) |
| carrier / speed / Doppler | 2.1 GHz / 50 km/h / 97.3 Hz, Jakes spectrum, 50 paths per tap |
| VehA | built-in `'VehicularA'` (ITU-R M.1225), RMS delay spread 0.37 us |
| SUI5 | self-defined PDP: 0 / 4 / 10 us, 0 / -5 / -10 dB (Rayleigh), RMS delay spread 2.84 us |
| ground truth | `GetTransferFunction` at the centre of each OFDM symbol |
| realisations | 40 000 independent sub-frames per model |

Usage (from this folder):

```matlab
sim = '../../supporting/5GLLSimulator';
generate_vienna_channels(sim, '../data', 40000, 2026, {'VehA'}, '')
generate_vienna_channels(sim, '../data', 13334, 2027, {'SUI5'}, '_part1')   % SUI5 in 3 parallel chunks
generate_vienna_channels(sim, '../data', 13333, 2028, {'SUI5'}, '_part2')
generate_vienna_channels(sim, '../data', 13333, 2029, {'SUI5'}, '_part3')
```

Output: `../data/vienna_<model>[suffix].mat` with `H` (N x 72 x 14, complex single) and `meta`.
The Python side (`channel.source: mat`, files in `channel.mat_files`) concatenates the chunks and
normalises the power to E|H|^2 = 1.
