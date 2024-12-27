import os
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot
import mne
from mne.preprocessing.nirs import (
    optical_density,
    temporal_derivative_distribution_repair,
)

# Import data
# -----------
plt.rcParams['font.family'] = 'Times New Roman'
# 加载 fNIRS 数据
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data().resample(3, npad="auto")

# 将光强数据转换为光学密度
raw_od = optical_density(raw_intensity)

# 检查 raw_od 的 info 字段，确保包含了光源频率
# 通常在 raw_od.info 中会包含通道的信息，例如波长（如果波长信息已经包含在 raw_od 数据中）
for ch in raw_od.info['chs']:
    print(ch)

# 或者打印 info，查找是否有包含 'wavelengths' 的信息
print(raw_od.info)

# 如果 raw_od.info 中没有 'nirs_info'，可以手动推测通道的波长
# 这里假设通道名中包含 '760' 和 '850' 来确定波长
# 选择包含 760 和 850 的通道
raw_od_multi_channel = raw_od.pick_channels([ch for ch in raw_od.info['ch_names'] if '760' in ch or '850' in ch])

# 打印选择的通道，确认是否有两个波长
print(raw_od_multi_channel.info)

# 添加注释（例如运动事件）
new_annotations = mne.Annotations(
    [31, 187, 317], [8, 8, 8], ["Movement", "Movement", "Movement"]
)
raw_od_multi_channel.set_annotations(new_annotations)

# 获取数据的最大值和最小值，用于设置绘图的坐标范围
data = raw_od_multi_channel.get_data()
time_points = raw_od_multi_channel.times  # 获取时间点
max_value = data.max()
min_value = data.min()

# 绘制原始数据的图像（包含两个波长的通道）
plt.figure(figsize=(10, 6))  # 设置图片大小
plt.plot(time_points, data[0, :], label="760 nm")  # 绘制 760 nm 的数据
plt.plot(time_points, data[1, :], label="850 nm")  # 绘制 850 nm 的数据
plt.title("Original Data")
plt.xlabel("Time (s)")
plt.ylabel("Optical Density")
plt.legend()

# 调整横纵坐标范围
plt.xlim([time_points[0], time_points[-1]])  # 设置横坐标范围
plt.ylim([min_value - 0.1, max_value + 0.1])  # 设置纵坐标范围，并加一些间距

plt.show()  # 显示图像

# 给数据添加伪影（例如，一个尖峰伪影和基线漂移）
corrupted_data = raw_od_multi_channel.get_data()

# 添加尖峰伪影（例如在 100 秒处）
corrupted_data[0, 298:302] = corrupted_data[0, 298:302] - 0.06  # 760 nm 的尖峰

# 添加基线漂移伪影（例如在 200 秒到 400 秒之间）
corrupted_data[1, 450:750] = corrupted_data[1, 450:750] + 0.03  # 850 nm 的基线漂移

# 创建包含伪影的 raw 数据对象
corrupted_od_multi_channel = mne.io.RawArray(
    corrupted_data, raw_od_multi_channel.info, first_samp=raw_od_multi_channel.first_samp
)

# 添加伪影注释
new_annotations.append([95, 145, 245], [10, 10, 10], ["Spike", "Baseline", "Baseline"])
corrupted_od_multi_channel.set_annotations(new_annotations)

# 获取伪影数据的最大值和最小值
corrupted_data = corrupted_od_multi_channel.get_data()
max_value_corrupted = corrupted_data.max()
min_value_corrupted = corrupted_data.min()

# 绘制伪影数据的图像（包含两个波长的通道）
plt.figure(figsize=(10, 6))  # 设置图片大小
plt.plot(time_points, corrupted_data[0, :], label="760 nm")  # 绘制 760 nm 的数据
plt.plot(time_points, corrupted_data[1, :], label="850 nm")  # 绘制 850 nm 的数据
plt.title("Corrupted Data")
plt.xlabel("Time (s)")
plt.ylabel("Optical Density")
plt.legend()

# 调整横纵坐标范围
plt.xlim([time_points[0], time_points[-1]])  # 设置横坐标范围
plt.ylim([min_value_corrupted - 0.1, max_value_corrupted + 0.1])  # 设置纵坐标范围，并加一些间距
plt.savefig('/Users/run/Desktop/DSP_Project/SB/fg11.pdf', format='pdf')
plt.show()  # 显示图像

# 使用时间导数分布修复来处理伪影
corrected_tddr_multi_channel = temporal_derivative_distribution_repair(corrupted_od_multi_channel)

# 获取修复后数据的最大值和最小值
corrected_data = corrected_tddr_multi_channel.get_data()
max_value_corrected = corrected_data.max()
min_value_corrected = corrected_data.min()

# 绘制修复后的数据图像（包含两个波长的通道）
plt.figure(figsize=(12, 8))  # 设置图片大小
plt.plot(time_points, corrected_data[0, :], label="760 nm")  # 绘制 760 nm 的数据
plt.plot(time_points, corrected_data[1, :], label="850 nm")  # 绘制 850 nm 的数据
plt.title("Corrected Data")
plt.xlabel("Time (s)")
plt.ylabel("Optical Density")
plt.legend()

# 调整横纵坐标范围
plt.xlim([time_points[0], time_points[-1]])  # 设置横坐标范围
plt.ylim([min_value_corrected - 0.1, max_value_corrected + 0.1])  # 设置纵坐标范围，并加一些间距

plt.show()  # 显示图像
