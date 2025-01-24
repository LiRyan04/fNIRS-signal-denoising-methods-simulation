import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.ndimage import gaussian_filter1d
from mne.preprocessing.nirs import optical_density

# 读取 fNIRS 数据
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().resample(3, npad="auto")

# 将信号转换为光密度
raw_od = optical_density(raw_intensity)

# 添加伪影 (尖峰伪影和基线漂移)
corrupted_data = raw_od.get_data()
corrupted_data[:, 298:302] -= 0.06  # 添加尖峰伪影
corrupted_data[:, 450:750] += 0.03  # 添加基线漂移

# 检查数据中的 NaN 或 Inf 值
if np.any(np.isnan(corrupted_data)) or np.any(np.isinf(corrupted_data)):
    print("Warning: Corrupted data contains NaN or Inf values.")
    corrupted_data = np.nan_to_num(corrupted_data)  # 将 NaN 转换为 0，Inf 转换为有限的数值

# 创建带有伪影的 RawArray
corrupted_od = mne.io.RawArray(corrupted_data, raw_od.info, first_samp=raw_od.first_samp)

# 改进的 MDF 函数
def multivariate_disturbance_filtering_improved(corrupted_data, order=(5, 5)):
    """
    改进版多变量扰动滤波器 (MDF)，增强对基线漂移的校正。
    """
    n_channels, n_samples = corrupted_data.shape
    filtered_data = np.zeros_like(corrupted_data)

    for i in range(n_channels):
        y = corrupted_data[i]

        # 数据归一化，避免标准差为零
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std == 0:  # 防止标准差为零
            y_normalized = y - y_mean
        else:
            y_normalized = (y - y_mean) / y_std

        # 基线漂移估计（低频分量）
        baseline_drift = gaussian_filter1d(y_normalized, sigma=50)  # 提取低频分量
        y_detrended = y_normalized - baseline_drift  # 移除基线漂移

        # 初始化参数
        params_init = np.random.uniform(-0.05, 0.05, order[0] + order[1])  # 缩小初始参数范围

        # 优化 ARMA 参数
        try:
            result = minimize(
                log_likelihood, params_init, args=(y_detrended, order, 0.01),
                method="L-BFGS-B",
                bounds=[(-0.9, 0.9)] * len(params_init),
                options={"disp": False, "maxiter": 500}
            )

            if not result.success:
                print(f"Channel {i}: Optimization failed. Using original data.")
                filtered_data[i] = y  # 未滤波
                continue

            params_opt = result.x
            a = np.r_[1, -params_opt[:order[0]]]
            b = np.r_[1, params_opt[order[0]:]]

            # 滤波并还原归一化数据
            filtered_normalized = lfilter(b, a, y_detrended)
            filtered_data[i] = filtered_normalized * y_std + y_mean + baseline_drift * y_std

            # 后处理 - 高斯平滑
            filtered_data[i] = gaussian_filter1d(filtered_data[i], sigma=3)

        except Exception as ex:
            print(f"Channel {i}: Exception occurred: {ex}")
            filtered_data[i] = y

    return filtered_data

# 对数似然函数
def log_likelihood(params, y, order, regularization_weight=0.01):
    """
    对数似然函数，用于优化 ARMA 模型参数。
    """
    p, q = order
    a = np.r_[1, -params[:p]]  # AR 系数
    b = np.r_[1, params[p:p + q]]  # MA 系数

    # 检查滤波器稳定性
    if np.any(np.abs(np.roots(a)) >= 0.95) or np.any(np.abs(np.roots(b)) >= 0.95):
        return np.inf  # 滤波器不稳定

    # 计算残差
    e = lfilter(b, a, y)  # 滤波后的残差

    # 检查数值有效性
    if np.any(np.isnan(e)) or np.any(np.isinf(e)):
        print(f"Invalid value encountered in residuals: {e}")  # 输出出错数据
        return np.inf

    # 对数似然
    ll = -0.5 * len(e) * np.log(2 * np.pi) - 0.5 * np.sum(e ** 2)
    reg = regularization_weight * np.sum(params ** 2)  # 加入正则化项
    return -(ll - reg)  # 返回负对数似然（供优化器最小化）

# 应用改进的 MDF 滤波
filtered_data_improved = multivariate_disturbance_filtering_improved(corrupted_data, order=(5, 5))

# 计算 CNR, SNR 和 MSE
def calculate_metrics(original_data, corrupted_data, filtered_data):
    """
    计算信号的 CNR, SNR 和 MSE。
    """
    mse = np.mean((original_data - filtered_data) ** 2)
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - filtered_data) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)

    # 计算 CNR（对每个通道单独计算后求均值）
    cnr_values = []
    for i in range(original_data.shape[0]):
        signal_range = np.ptp(original_data[i])
        noise_range = np.ptp(original_data[i] - filtered_data[i])
        cnr = 10 * np.log10(signal_range / noise_range) if noise_range > 0 else np.inf
        cnr_values.append(cnr)

    cnr = np.mean(cnr_values)
    return cnr, snr, mse

# 计算并打印指标
cnr, snr, mse = calculate_metrics(raw_od.get_data(), corrupted_data, filtered_data_improved)
print(f"CNR: {cnr:.2f} dB")
print(f"SNR: {snr:.2f} dB")
print(f"MSE: {mse:.6f}")

# 绘图函数
# 绘图函数
def plot_separate(raw_data, corrupted_data, filtered_data, channel_idx=0):
    """
    分别绘制原始信号、伪影信号和滤波后的信号。
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 原始信号
    axes[0].plot(raw_data[channel_idx], label="Original Signal", color="blue", alpha=0.7)
    axes[0].set_title(f"Channel {channel_idx + 1} - Original Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")

    # 加噪声的信号
    axes[1].plot(corrupted_data[channel_idx], label="Corrupted Signal", color="orange", alpha=0.7)
    axes[1].set_title(f"Channel {channel_idx + 1} - Corrupted Signal")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")

    # 去噪后的信号
    axes[2].plot(filtered_data[channel_idx], label="Filtered Signal (Improved MDF)", color="green", alpha=0.7)
    axes[2].set_title(f"Channel {channel_idx + 1} - Filtered Signal")
    axes[2].set_xlabel("Time (samples)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# 绘制第 1 个通道的信号
plot_separate(raw_od.get_data(), corrupted_data, filtered_data_improved, channel_idx=0)
