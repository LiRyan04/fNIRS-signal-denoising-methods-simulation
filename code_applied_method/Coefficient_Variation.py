import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'Times New Roman'

# Function to convert raw intensity to optical density
def optical_density(raw_intensity):
    """
    Convert raw intensity to optical density (OD) using Beer-Lambert law.
    """
    raw_intensity_data = raw_intensity.get_data()  # Get the actual data from the Raw object
    I0 = np.mean(raw_intensity_data, axis=1, keepdims=True)  # Baseline intensity (mean over time)
    return -np.log(raw_intensity_data / I0)

# Coefficient of Variation (CV) Filtering function
def cv_filtering(data, threshold=0.2):
    """
    Apply coefficient of variation (CV) filtering to remove high-variability regions (artifact regions).

    :param data: Input data (n_channels, n_samples)
    :param threshold: CV threshold above which regions will be considered as noise and filtered
    :return: Data with high-variance regions filtered
    """
    n_channels, n_samples = data.shape
    filtered_data = data.copy()

    # Iterate over each channel
    for i in range(n_channels):
        # Calculate the coefficient of variation (CV) for the current channel
        mean = np.mean(data[i, :])
        std_dev = np.std(data[i, :])
        cv = std_dev / mean if mean != 0 else 0

        # If CV is greater than the threshold, consider it as an artifact region
        if cv > threshold:
            # Apply a simple smoothing technique to filter out the high-variance region (artifact)
            filtered_data[i, :] = np.convolve(data[i, :], np.ones(5)/5, mode='same')  # Simple moving average filter

    return filtered_data

# Load fNIRS data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# Convert to optical density
raw_od = optical_density(raw_intensity)

# Simulate noise signal
np.random.seed(42)  # For reproducibility
noise = np.random.normal(loc=0, scale=0.01, size=raw_od.shape)  # Gaussian noise

# Create corrupted data (add artifacts + noise)
corrupted_data = raw_od + noise  # Combine signal with noise
corrupted_data[:, 298:302] = corrupted_data[:, 298:302] - 0.06  # Add spike artifact
corrupted_data[:, 450:750] = corrupted_data[:, 450:750] + 0.03  # Add baseline drift

# Create RawArray with corrupted data
corrupted_od = mne.io.RawArray(corrupted_data, raw_intensity.info, first_samp=raw_intensity.first_samp)

# Apply CV-based filtering
cv_filtered_data = cv_filtering(corrupted_data, threshold=0.2)

# Create RawArray with filtered data
filtered_od = mne.io.RawArray(cv_filtered_data, raw_intensity.info, first_samp=raw_intensity.first_samp)

# Set consistent y-axis limits for plotting
y_min = min(np.min(raw_od[0, :]), np.min(corrupted_data[0, :]), np.min(cv_filtered_data[0, :]))
y_max = max(np.max(raw_od[0, :]), np.max(corrupted_data[0, :]), np.max(cv_filtered_data[0, :]))

# Time axis for plotting
fs = raw_intensity.info['sfreq']  # Sampling frequency
time_axis = np.arange(raw_od.shape[1]) / fs

# Plot raw, corrupted, and CV-filtered data
plt.figure(figsize=(12, 10))

# Raw data (Original Signal)
plt.subplot(3, 1, 1)
plt.plot(time_axis, raw_od[0, :], label="Raw Data", color='blue', linewidth=1.5)
plt.title("Raw fNIRS Data")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

# Corrupted data with artifacts and noise
plt.subplot(3, 1, 2)
plt.plot(time_axis, corrupted_data[0, :], label="Corrupted Data", color='red', linewidth=1.5)
plt.axvspan(time_axis[298], time_axis[302], color='grey', alpha=0.3, label="Spike Artifact")
plt.axvspan(time_axis[450], time_axis[750], color='green', alpha=0.3, label="Baseline Drift")
plt.title("Corrupted fNIRS Data (with artifacts and noise)")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

# CV-filtered data
plt.subplot(3, 1, 3)
plt.plot(time_axis, cv_filtered_data[0, :], label="CV Filtered Data", color='purple', linewidth=1.5)
plt.title("Filtered fNIRS Data (CV Filtering)")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

plt.tight_layout()
plt.show()

# Calculate SNR, SME, and CNR

# SNR (Signal-to-Noise Ratio) - Compare the signal power and noise power
signal_power = np.mean(raw_od ** 2)  # Power of the raw signal
noise_power = np.mean(noise ** 2)    # Power of the noise
SNR = 10 * np.log10(signal_power / noise_power)

# SME (Signal-to-Mean Error) - Mean square error between raw and filtered data
sme = np.mean((raw_od - cv_filtered_data) ** 2)

# CNR (Contrast-to-Noise Ratio) - Contrast between signal and noise
signal_contrast = np.mean(np.abs(raw_od - corrupted_data))  # Contrast between original and corrupted signal
CNR = 10 * np.log10(signal_contrast / noise_power)

# Output the metrics
print(f"SNR (Signal-to-Noise Ratio): {SNR:.2f} dB")
print(f"SME (Signal-to-Mean Error): {sme:.5f}")
print(f"CNR (Contrast-to-Noise Ratio): {CNR:.2f} dB")
# 记录程序开始的时间
start_time = time.time()

# 你的代码
for i in range(1000000):
    pass

# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")