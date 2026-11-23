"""ChannelNet reproduction package (Soltani et al., "Deep Learning-Based Channel Estimation", 2019).

Modules
-------
config          YAML configuration loading / CLI overrides
utils           seeding, logging, device selection, small helpers
ofdm            LTE-like OFDM grid, lattice pilot patterns, OFDM (de)modulator, LS pilot observation
channel_models  VehA / SUI5 tapped-delay-line fading channels -> 72x14 time-frequency response
dataset         channel generation/caching, splits, noisy LS observations, image conversion
estimators      conventional estimators: LS + 2-D interpolation, robust / estimated / ideal LMMSE
models          SRCNN, DnCNN and the cascaded ChannelNet
trainer         generic stage-wise trainer (early stopping, LR schedule, checkpoints, loss history)
evaluation      MSE-vs-SNR / MSE-vs-pilots evaluation of all methods
plotting        all figures of the report
"""
