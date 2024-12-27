import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing.nirs import optical_density
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

plt.rcParams['font.family'] = 'Times New Roman'

# Load fNIRS data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# Convert signal to optical density
raw_od = optical_density(raw_intensity)

# Simulate corrupted data (e.g., with artifacts)
corrupted_data = raw_od.get_data()  # Get raw data
corrupted_data[:, 298:302] = corrupted_data[:, 298:302] - 0.06  # Add spike artifact
corrupted_data[:, 450:750] = corrupted_data[:, 450:750] + 0.03  # Add baseline drift

# Create a new RawArray with corrupted data
corrupted_od = mne.io.RawArray(corrupted_data, raw_od.info, first_samp=raw_od.first_samp)

# Savitzky-Golay Filter Function
def apply_sg_filter(data, window_length=51, polyorder=3):
    """Apply the Savitzky-Golay filter to smooth the signal."""
    return savgol_filter(data, window_length, polyorder, axis=1)

# Apply the filter to the corrupted data
window_length = 51  # Window length (must be odd)
polyorder = 3  # Polynomial order
smoothed_data = apply_sg_filter(corrupted_data, window_length, polyorder)

# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original_signal, filtered_signal):
    """Calculate the Signal-to-Noise Ratio (SNR)."""
    signal_variance = np.var(original_signal)
    noise_variance = np.var(original_signal - filtered_signal)
    snr = signal_variance / noise_variance
    return snr

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_signal, filtered_signal):
    """Calculate the Mean Squared Error (MSE) between original and filtered signals."""
    mse = mean_squared_error(original_signal, filtered_signal)
    return mse

# Calculate SNR and MSE for the first channel 
snr_value = calculate_snr(corrupted_data[0], smoothed_data[0])
mse_value = calculate_mse(corrupted_data[0], smoothed_data[0])

print(f"Signal-to-Noise Ratio (SNR): {snr_value:.4f}")
print(f"Mean Squared Error (MSE): {mse_value:.4f}")

# Function to plot signals
def plot_signals(raw_data, filtered_data):
    """Plot raw and filtered signals."""
    plt.figure(figsize=(12, 8))

    # Raw (corrupted) data
    plt.subplot(2, 1, 1)
    plt.plot(raw_data, label="Corrupted Signal", color='red', linewidth=1.5)
    plt.axvspan(298, 302, color='gray', alpha=0.5, label='Spike Artifact', linewidth=1)
    plt.axvspan(450, 750, color='yellow', alpha=0.3, label='Baseline Drift', linewidth=1)
    plt.title("Raw fNIRS Data (with Artifacts)")
    plt.xlabel("Samples")
    plt.ylabel("Optical Density (OD)")
    plt.legend()

    # Smoothing (filtered) data
    plt.subplot(2, 1, 2)
    plt.plot(filtered_data, label="Filtered Signal", linewidth=1.5)
    plt.title("Filtered fNIRS Data (Savitzky-Golay)")
    plt.xlabel("Samples")
    plt.ylabel("Optical Density (OD)")



    plt.tight_layout()
    plt.show()

# Plot the original and filtered signals for comparison
plot_signals(corrupted_data[0], smoothed_data[0])

# Calculate and print the SNR and MSE for all channels
snr_values = []
mse_values = []
for i in range(corrupted_data.shape[0]):  # Loop over channels
    snr_values.append(calculate_snr(corrupted_data[i], smoothed_data[i]))
    mse_values.append(calculate_mse(corrupted_data[i], smoothed_data[i]))

print(f"Average SNR: {np.mean(snr_values):.4f}")
print(f"Average MSE: {np.mean(mse_values):.4f}")

# Function to calculate Contrast-to-Noise Ratio (CNR)
def calculate_cnr(original_signal, filtered_signal):
    """Calculate the Contrast-to-Noise Ratio (CNR)."""
    active_signal = original_signal[300:350]  # Example range for "active" signal
    baseline_signal = original_signal[0:50]  # Example range for baseline 
    contrast = np.mean(active_signal) - np.mean(baseline_signal)
    noise = np.std(original_signal - filtered_signal)
    cnr = np.abs(contrast) / noise
    return cnr

# Calculate CNR for the first channel
cnr_value = calculate_cnr(corrupted_data[0], smoothed_data[0])
print(f"Contrast-to-Noise Ratio (CNR): {cnr_value:.4f}")
