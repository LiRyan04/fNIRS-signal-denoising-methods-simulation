import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing.nirs import optical_density
from scipy.interpolate import interp1d
from scipy.stats import zscore
import time

# 记录程序开始的时间
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


# Function to calculate moving standard deviation (MSD) for artifact identification
def moving_std(signal, window_size):
    """
    Calculate the moving standard deviation of a signal.
    """
    return np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same') -
                   np.convolve(signal, np.ones(window_size) / window_size, mode='same') ** 2)


# Function to apply linear interpolation for baseline correction
def apply_linear_interpolation(signal, artifact_indices):
    """
    Apply linear interpolation to correct baseline drift.
    Only interpolate over regions that have been identified as artifacts.
    """
    time = np.arange(len(signal))

    # Mask the artifact regions (where artifact_indices is True)
    mask = ~artifact_indices
    if np.sum(mask) < 3:
        return signal  # If there are very few valid points, return original signal

    # Perform linear interpolation on the signal
    interp_func = interp1d(time[mask], signal[mask], kind='linear', fill_value='extrapolate')

    # Reconstruct the signal with baseline corrected
    corrected_signal = signal.copy()
    corrected_signal[artifact_indices] = interp_func(time[artifact_indices])

    return corrected_signal


# Function to remove spike artifacts using a simple form of Loess (Local Weighted Regression)
def remove_spikes(signal, window_size=5, bandwidth=0.15):
    """
    Remove spike artifacts using a form of local weighted regression (Loess).
    """
    smoothed_signal = signal.copy()
    half_window = window_size // 2
    for i in range(half_window, len(signal) - half_window):
        # Define local window around the point
        window = signal[i - half_window:i + half_window + 1]
        # Calculate the weight for each neighboring point
        weight = (1 - np.abs(window - signal[i]) / bandwidth) ** 3
        weight[weight < 0] = 0  # Ensure non-negative weights
        # Perform weighted regression
        smoothed_signal[i] = np.dot(weight, window) / weight.sum()
    return smoothed_signal


# 计算并打印标准差
window_size = 50  # 这个值可以调试，避免过度平滑
std_dev = moving_std(corrupted_data[0], window_size)

# 通过标准差检测伪影
threshold = np.percentile(std_dev, 90)  # 调整阈值，避免误判
artifact_indices = std_dev > threshold  # 识别伪影区域

# 进行尖峰修正（Loess平滑）
final_corrected_signal = remove_spikes(corrupted_data[0], window_size=10, bandwidth=0.2)

# 进行基线修正（线性插值）
final_corrected_signal = apply_linear_interpolation(final_corrected_signal, artifact_indices)


# 计算 SNR, CNR, MSE

def calculate_snr(signal, noise_signal):
    """计算信噪比(SNR)"""
    signal_var = np.var(signal)
    noise_var = np.var(noise_signal)
    snr = signal_var / noise_var
    return snr

def calculate_cnr(signal, noise_signal):
    """计算对比噪声比(CNR)"""
    signal_std = np.std(signal)
    noise_std = np.std(noise_signal)
    cnr = signal_std / noise_std
    return cnr

def calculate_mse(original_signal, processed_signal):
    """计算均方误差(MSE)"""
    mse = np.mean((original_signal - processed_signal) ** 2)
    return mse

# 获取原始信号和清理后的信号
raw_signal = raw_od.get_data()[0]  # 原始信号（无噪声）
clean_signal = raw_signal - corrupted_data[0]  # 假设清理后的信号即为原始信号（无噪声）
# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")

# 计算SNR、CNR和MSE
snr_value = calculate_snr(raw_signal, clean_signal)
cnr_value = calculate_cnr(raw_signal, clean_signal)
mse_value = calculate_mse(raw_signal, final_corrected_signal)

print(f"SNR: {snr_value:.2f}")
print(f"CNR: {cnr_value:.2f}")
print(f"MSE: {mse_value:.2f}")

# 可视化仅滤波后的信号
def plot_filtered_signal(corrected_data, title):
    plt.figure(figsize=(10, 6))

    # Plot corrected signal
    plt.plot(corrected_data)
    plt.title(f"{title} - Processed Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Set y-axis limit for better visualization
    plt.tight_layout()
    plt.show()


# Plot the filtered (corrected) signal
plot_filtered_signal(final_corrected_signal, "Linear Interpolation & Loess")

