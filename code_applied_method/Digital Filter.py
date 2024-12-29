import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import time
# 记录程序开始的时间
start_time = time.time()

# 你的代码
for i in range(1000000):
    pass
# Function to apply a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

# Function to remove DC offset
def remove_dc_offset(data):
    return data - np.mean(data, axis=1, keepdims=True)

# Function to convert raw intensity to optical density
def optical_density(raw_intensity):
    """Convert raw intensity to optical density (OD) using Beer-Lambert law."""
    I0 = np.mean(raw_intensity, axis=1, keepdims=True)  # Baseline intensity (mean over time)
    return -np.log(raw_intensity / I0)

# Load fNIRS data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# Convert to optical density
raw_od = optical_density(raw_intensity.get_data())

# Simulate corrupted data (e.g., with artifacts)
corrupted_data = raw_od.copy()
corrupted_data[:, 298:302] += 0.06  # Add spike artifact
corrupted_data[:, 450:750] += 0.03  # Add baseline drift

# Remove DC offset from corrupted data
dc_removed_data = remove_dc_offset(corrupted_data)

# Apply bandpass filter to the DC-removed data
fs = raw_intensity.info['sfreq']  # Sampling frequency
lowcut = 0.01  # Low cutoff frequency (Hz)
highcut = 0.5  # High cutoff frequency (Hz)
filtered_data = bandpass_filter(dc_removed_data, lowcut, highcut, fs)

# Set consistent y-axis limits for plotting
y_min = min(np.min(raw_od[0, :]), np.min(corrupted_data[0, :]), np.min(filtered_data[0, :]))
y_max = max(np.max(raw_od[0, :]), np.max(corrupted_data[0, :]), np.max(filtered_data[0, :]))

# Time axis for plotting
time_axis = np.arange(raw_od.shape[1]) / fs
# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")
# Plot the raw, corrupted, and filtered data
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

# Corrupted data with artifacts
plt.subplot(3, 1, 2)
plt.plot(time_axis, corrupted_data[0, :], label="Corrupted Data (with artifacts)", color='red', linewidth=1.5)
plt.axvspan(time_axis[298], time_axis[302], color='gray', alpha=0.5, label="Spike Artifact")
plt.axvspan(time_axis[450], time_axis[750], color='yellow', alpha=0.3, label="Baseline Drift")
plt.title("Corrupted fNIRS Data (with artifacts)")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

# Filtered (denoised) data
plt.subplot(3, 1, 3)
plt.plot(time_axis, filtered_data[0, :], label="Filtered Data (Denoised)", color='green', linewidth=1.5)
plt.title("Filtered fNIRS Data (Denoised)")
plt.xlabel("Time (seconds)")
plt.ylabel("Optical Density (OD)")
plt.legend()
plt.xlim([time_axis[0], time_axis[-1]])
plt.ylim([y_min, y_max])

plt.tight_layout()
plt.savefig("fnirs_denoising_revised.png")
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
