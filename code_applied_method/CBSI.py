import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing.nirs import optical_density
from scipy.stats import pearsonr
from pywt import wavedec, waverec
import time

# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

# 开始计时
start_time = time.time()

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

# 使用 Pearson 相关系数计算相关性矩阵（PCC）
def compute_pcc_matrix(data):
    n_channels = data.shape[0]
    pcc_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            r, _ = pearsonr(data[i, :], data[j, :])  # 计算两个信号的相关系数
            pcc_matrix[i, j] = r
            pcc_matrix[j, i] = r  # 相关矩阵是对称的
    return pcc_matrix

# 计算PCC矩阵
pcc_matrix = compute_pcc_matrix(corrupted_data)

# 绘制PCC矩阵
plt.figure(figsize=(10, 7))
plt.imshow(pcc_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('PCC Correlation Matrix')
plt.show()

# 使用小波去噪方法
def wavelet_denoising(signal, wavelet='db4', level=4):
    coeffs = wavedec(signal, wavelet, level=level)  # 小波分解
    # 设置噪声系数为零
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    denoised_signal = waverec(coeffs, wavelet)  # 小波重构
    return denoised_signal

# 对每个通道应用小波去噪
denoised_data = np.apply_along_axis(wavelet_denoising, 1, corrupted_data)

# 绘制带伪影的数据和去噪后的数据波形
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 绘制带伪影的信号
axs[0].plot(corrupted_od.times, corrupted_od.get_data()[0], label="Signal with Motion Artifacts")
axs[0].set_title("Original Signal with Motion Artifacts")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude (AU)")
axs[0].legend()

# 绘制去噪后的信号
axs[1].plot(corrupted_od.times, denoised_data[0], label="Denoised Signal", color='g')
axs[1].set_title("Signal After Denoising")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Amplitude (AU)")
axs[1].legend()

plt.tight_layout()
plt.show()

# 使用CBSI方法
def cbsimethod(HbO, HbR):
    alpha = np.std(HbO) / np.std(HbR)  # 计算α
    TNS = 0.5 * (HbO + alpha * HbR)  # 真噪声信号
    TFS = 0.5 * (HbO - alpha * HbR)  # 真特征信号
    return TNS, TFS

# 假设HbO和HbR信号已被提取，这里仅为演示
HbO = denoised_data[0]  # 取第一个通道作为氧合血红蛋白信号
HbR = denoised_data[1]  # 取第二个通道作为脱氧血红蛋白信号

# 应用CBSI方法
TNS, TFS = cbsimethod(HbO, HbR)

# Output program runtime
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# 绘制CBSI处理后的信号
plt.figure(figsize=(10, 6))
plt.plot(TNS, label="True Noise Signal (TNS)")
plt.plot(TFS, label="True Feature Signal (TFS)")
plt.legend()
plt.title("Signal After CBSI Method")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (AU)")
plt.show()

# 噪声为原始信号与去噪信号的差异
noise = corrupted_data - denoised_data  # 噪声是带伪影部分与去噪信号的差异

# SNR 计算
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# CNR 计算
def calculate_cnr(signal, background, noise):
    signal_mean = np.mean(signal)
    background_mean = np.mean(background)
    noise_std = np.std(noise)
    return (signal_mean - background_mean) / noise_std

# MSE 计算
def calculate_mse(original, denoised):
    return np.mean((original - denoised) ** 2)

# 计算每个通道的 SNR, CNR 和 MSE
snr_values = np.apply_along_axis(calculate_snr, 1, denoised_data, noise=noise)
cnr_values = np.apply_along_axis(calculate_cnr, 1, denoised_data, background=np.zeros_like(denoised_data), noise=noise)
mse_values = np.array([calculate_mse(corrupted_data[i], denoised_data[i]) for i in range(corrupted_data.shape[0])])


# Calculate average SNR, CNR, and MSE
average_snr = np.mean(snr_values)
average_cnr = np.mean(cnr_values)
average_mse = np.mean(mse_values)


# Print average metrics
print(f"Average SNR: {average_snr:.2f}")
print(f"Average CNR: {average_cnr:.2f}")
print(f"Average MSE: {average_mse:.8f}")