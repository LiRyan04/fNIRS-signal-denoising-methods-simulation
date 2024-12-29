import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing.nirs import optical_density
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
start_time = time.time()

# 你的代码
for i in range(1000000):
    pass
# 读取 fNIRS 数据
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# 将信号转换为光密度
raw_od = optical_density(raw_intensity)

# 示例：给信号添加伪影
corrupted_data = raw_od.get_data()  # 获取原始数据
corrupted_data[:, 298:302] = corrupted_data[:, 298:302] - 0.06  # 添加尖峰伪影
corrupted_data[:, 450:750] = corrupted_data[:, 450:750] + 0.03  # 添加基线漂移

# 创建带有伪影的新RawArray
corrupted_od = mne.io.RawArray(corrupted_data, raw_od.info, first_samp=raw_od.first_samp)


# Savitzky-Golay Filter Function
def apply_sg_filter(data, window_length=51, polyorder=3):
    """
    Apply the Savitzky-Golay filter to smooth the signal.
    """
    return savgol_filter(data, window_length, polyorder, axis=1)


# Apply the filter to the corrupted data
window_length = 51  # Window length (must be odd)
polyorder = 3  # Polynomial order
smoothed_data = apply_sg_filter(corrupted_data, window_length, polyorder)
# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")

# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original_signal, filtered_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR).
    """
    signal_variance = np.var(original_signal)
    noise_variance = np.var(original_signal - filtered_signal)
    snr = signal_variance / noise_variance
    return snr


# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_signal, filtered_signal):
    """
    Calculate the Mean Squared Error (MSE) between original and filtered signals.
    """
    mse = mean_squared_error(original_signal, filtered_signal)
    return mse


# Function to calculate Contrast-to-Noise Ratio (CNR)
def calculate_cnr(original_signal, filtered_signal, noise_window=50):
    """
    Calculate the Contrast-to-Noise Ratio (CNR).
    """
    # Calculate the signal contrast (mean of the signal part)
    signal_part = original_signal
    signal_contrast = np.mean(signal_part)

    # Estimate noise as the difference between original and filtered signals
    noise = original_signal - filtered_signal
    noise_std = np.std(noise[-noise_window:])  # Noise standard deviation in the last part (adjustable)

    cnr = signal_contrast / noise_std
    return cnr


# Calculate SNR, MSE, and CNR for the first channel
snr_value = calculate_snr(corrupted_data[0], smoothed_data[0])
mse_value = calculate_mse(corrupted_data[0], smoothed_data[0])
cnr_value = calculate_cnr(corrupted_data[0], smoothed_data[0])

print(f"Signal-to-Noise Ratio (SNR): {snr_value:.4f}")
print(f"Mean Squared Error (MSE): {mse_value:.4f}")
print(f"Contrast-to-Noise Ratio (CNR): {cnr_value:.4f}")


# Visualize the results (raw vs. filtered signal)
def plot_signals(raw_data, filtered_data, title):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(raw_data)
    plt.title("Raw Signal with Artifacts")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(filtered_data)
    plt.title(f"{title} - Processed Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


# Plot the original and filtered signals for comparison
plot_signals(corrupted_data[0], smoothed_data[0], "Savitzky-Golay Filter")
