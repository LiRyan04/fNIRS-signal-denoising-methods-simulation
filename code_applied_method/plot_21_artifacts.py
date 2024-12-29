import os
import matplotlib.pyplot as plt
import mne
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from mne.preprocessing.nirs import (
    optical_density,
    temporal_derivative_distribution_repair,
)
start_time = time.time()

# 你的代码
for i in range(1000000):
    pass
# 计算SNR
def calculate_snr(signal, noise):
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# 计算CNR
def calculate_cnr(signal, noise):
    signal_mean = np.mean(signal)
    noise_std = np.std(noise)
    cnr = np.abs(signal_mean) / noise_std
    return cnr

# 计算MSE
def calculate_mse(original, predicted):
    return mean_squared_error(original, predicted)

# 导入数据
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data().resample(3, npad="auto")
raw_od = optical_density(raw_intensity)

# 设置注释
new_annotations = mne.Annotations(
    [31, 187, 317], [8, 8, 8], ["Movement", "Movement", "Movement"]
)
raw_od.set_annotations(new_annotations)
# 记录程序结束的时间
end_time = time.time()

# 计算并输出程序运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time} 秒")
# 绘制原始数据
raw_od.plot(n_channels=15, duration=400, show_scrollbars=False)
plt.show()  # 添加这行以保持图表显示

# 添加人工伪影
corrupted_data = raw_od.get_data()
corrupted_data[:, 298:302] = corrupted_data[:, 298:302] - 0.06
corrupted_data[:, 450:750] = corrupted_data[:, 450:750] + 0.03
corrupted_od = mne.io.RawArray(
    corrupted_data, raw_od.info, first_samp=raw_od.first_samp
)
new_annotations.append([95, 145, 245], [10, 10, 10], ["Spike", "Baseline", "Baseline"])
corrupted_od.set_annotations(new_annotations)

# 绘制腐败数据
corrupted_od.plot(n_channels=15, duration=400, show_scrollbars=False)
plt.show()  # 添加这行以保持图表显示

# 应用时域导数分布修复（TDDR）
corrected_tddr = temporal_derivative_distribution_repair(corrupted_od)

# 绘制修复后的数据
corrected_tddr.plot(n_channels=15, duration=400, show_scrollbars=False)
plt.show()  # 添加这行以保持图表显示

# 计算SNR、CNR和MSE
original_data = raw_od.get_data()  # 获取原始数据
corrupted_data = corrupted_od.get_data()  # 获取腐败数据

# 计算噪声部分，假设噪声为腐败数据与原始数据的差异
noise = corrupted_data - original_data


# 计算SNR、CNR和MSE
snr = calculate_snr(original_data, noise)
cnr = calculate_cnr(original_data, noise)
mse = calculate_mse(original_data, corrupted_data)

# 输出SNR、CNR和MSE
print(f"SNR: {snr:.2f} dB")
print(f"CNR: {cnr:.2f}")
print(f"MSE: {mse:.6f}")


