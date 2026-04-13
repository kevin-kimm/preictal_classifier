import mne

mne.set_log_level('WARNING')

raw = mne.io.read_raw_edf('data/siena-scalp-eeg-database-1.0.0/PN00/PN00-1.edf', preload=False)

print(raw.ch_names)
print(f'Sample rate: {raw.info["sfreq"]} Hz')
print(f'Duration: {raw.n_times / raw.info["sfreq"] / 60:.1f} minutes')