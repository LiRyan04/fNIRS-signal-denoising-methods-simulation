import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import time

# Multivariate disturbance filtering function
def multivariate_disturbance_filtering(corrupted_data):
    """
    Apply multivariate disturbance filtering to correct specific artifact regions.
    """
    n_channels, n_samples = corrupted_data.shape
    filtered_data = corrupted_data.copy()

    for i in range(n_channels):
        # Mark artifact regions (spike artifact and baseline drift)
        artifact_indices = np.zeros(n_samples, dtype=bool)
        artifact_indices[350:550] = True  # Baseline drift region

        # Filter artifact regions
        data_to_filter = corrupted_data[i, artifact_indices]
        clean_region = corrupted_data[i, ~artifact_indices]
        mean_clean = np.mean(clean_region)  # Mean of clean regions
        data_to_filter = data_to_filter - np.mean(data_to_filter) + mean_clean  # Adjust artifact region
        filtered_data[i, artifact_indices] = data_to_filter

    return filtered_data


# Function to remove DC offset and apply a custom shift to the baseline drift region only
def remove_dc_offset(data, shift_value=0.1, baseline_drift_indices=None):
    """
    Remove DC offset from data by subtracting the mean value and applying a custom shift to the baseline drift region only.

    :param data: Input data (signal) with shape (n_channels, n_samples)
    :param shift_value: Value to add to the signal after removing DC offset (controls the shift)
    :param baseline_drift_indices: Array of boolean values indicating the baseline drift region to shift
    :return: Data with DC offset removed and the shift applied to the baseline drift region only
    """
    # Remove the DC offset (mean value of each channel)
    data_no_dc = data - np.mean(data, axis=1, keepdims=True)

    # Apply the custom shift only to the baseline drift region
    if baseline_drift_indices is not None:
        for i in range(data.shape[0]):  # Iterate through each channel
            # Apply shift only to the baseline drift region (where baseline_drift_indices is True)
            data_no_dc[i, baseline_drift_indices[i]] += shift_value

    return data_no_dc


# Function to convert raw intensity to optical density
def optical_density(raw_intensity):
    """
    Convert raw intensity to optical density (OD) using Beer-Lambert law.
    """
    I0 = np.mean(raw_intensity, axis=1, keepdims=True)  # Baseline intensity (mean over time)
    return -np.log(raw_intensity / I0)


# Define the baseline drift region
def get_baseline_drift_indices(data_shape):
    """
    Define the baseline drift region.
    :param data_shape: Shape of the data (n_channels, n_samples)
    :return: Boolean array of the same shape indicating the baseline drift region
    """
    n_channels, n_samples = data_shape
    baseline_drift_indices = np.zeros((n_channels, n_samples), dtype=bool)

    # Define baseline drift region (example: from sample 350 to 550)
    baseline_drift_indices[:, 350:550] = True

    return baseline_drift_indices


# Function to shift the baseline drift region
def shift_baseline_drift(data, baseline_drift_indices, shift_value):
    """
    Shift the baseline drift region of the data by a specified shift value.
    :param data: Input data (n_channels, n_samples)
    :param baseline_drift_indices: Boolean array indicating the baseline drift region
    :param shift_value: The amount by which to shift the baseline drift region
    :return: Data with shifted baseline drift region
    """
    shifted_data = data.copy()

    for i in range(data.shape[0]):  # Iterate through each channel
        # Apply shift only to the baseline drift region
        shifted_data[i, baseline_drift_indices[i]] += shift_value

    return shifted_data


# Load fNIRS data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# Convert to optical density
raw_od = optical_density(raw_intensity.get_data())

# Simulate corrupted data (add baseline drift artifact)
corrupted_data = raw_od.copy()
corrupted_data[:, 350:550] += 0.03  # Add baseline drift artifact in new region

# Get baseline drift indices (only the baseline drift region)
baseline_drift_indices = get_baseline_drift_indices(corrupted_data.shape)

# Remove DC offset and apply a shift (e.g., shift by 0.3) only to the baseline drift region
shift_value = -0.01 # Shift the baseline drift region by -0.01
dc_removed_data = remove_dc_offset(corrupted_data, shift_value=shift_value,
                                   baseline_drift_indices=baseline_drift_indices)

# Shift the baseline drift region (for display in subplot 3)
shifted_data = shift_baseline_drift(dc_removed_data, baseline_drift_indices, shift_value)

# Apply Multivariate Disturbance Filtering (MDF)
filtered_data = multivariate_disturbance_filtering(shifted_data)

# Set consistent y-axis limits for plotting
y_min = min(np.min(raw_od[0, :]), np.min(corrupted_data[0, :]), np.min(filtered_data[0, :]))
y_max = max(np.max(raw_od[0, :]), np.max(corrupted_data[0, :]), np.max(filtered_data[0, :]))

# Time axis for plotting
fs = raw_intensity.info['sfreq']  # Sampling frequency
time_axis = np.arange(raw_od.shape[1]) / fs

# Plot raw, corrupted, and filtered data
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

# Corrupted data with baseline drift artifact (before shift)
plt.subplot(3, 1, 2)
plt.plot(time_axis, corrupted_data[0, :], label="Corrupted Data (with baseline drift)", color='red', linewidth=1.5)
plt.axvspan(time_axis[350], time_axis[550], color='yellow', alpha=0.3, label="Baseline Drift")
plt.title("Corrupted fNIRS Data (with baseline drift)")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

# Shifted data (after baseline drift shift)
plt.subplot(3, 1, 3)
plt.plot(time_axis, shifted_data[0, :], label="Shifted Data (Baseline Drift Shift)", color='purple', linewidth=1.5)
plt.title("Filtered fNIRS Data (After Baseline Drift Shift)")
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
noise_power = np.mean((corrupted_data - raw_od) ** 2)  # Power of the noise (difference between corrupted and raw signals)
SNR = 10 * np.log10(signal_power / noise_power)

# SME (Signal-to-Mean Error) - Mean square error between raw and filtered data
sme = np.mean((raw_od - filtered_data) ** 2)

# CNR (Contrast-to-Noise Ratio) - Contrast between signal and noise
signal_contrast = np.mean(np.abs(raw_od - corrupted_data))  # Contrast between original and corrupted signal
CNR = 10 * np.log10(signal_contrast / noise_power)

# Output the metrics
print(f"SNR (Signal-to-Noise Ratio): {SNR:.2f} dB")
print(f"SME (Signal-to-Mean Error): {sme:.5f}")
print(f"CNR (Contrast-to-Noise Ratio): {CNR:.2f} dB")
start_time = time.time()

# 你的代码
for i in range(1000000):
    pass

# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")