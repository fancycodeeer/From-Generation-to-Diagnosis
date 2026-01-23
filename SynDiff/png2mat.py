import os
import numpy as np
from PIL import Image
import h5py


def save_images_to_hdf5(image_folder, output_mat_file):
	# 获取文件夹中的所有 PNG 图像
	image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
	image_files.sort()  # 按照文件名排序

	images = []

	for image_file in image_files:
		image_path = os.path.join(image_folder, image_file)

		# 打开图像并将其转换为灰度模式，如果需要彩色图像，可以修改此行
		img = Image.open(image_path)
		img = img.convert('L')

		# 转换为 NumPy 数组并归一化到 [0, 1] 之间
		img_array = np.array(img) / 255.0

		# 将图像数据添加到列表中
		images.append(img_array)

	# 转换为 NumPy 数组并重新排列维度，符合 (#images, width, height, channels) 结构
	images = np.array(images)

	# 如果是彩色图像，我们将其转换为 (#images, width, height, channels)
	# 如果是灰度图像，则应该是 (#images, width, height)，此处可以根据实际情况调整
	# 保存为 .mat 文件
	# 使用 HDF5 保存数据
	with h5py.File(output_mat_file, 'w') as f:
		f.create_dataset('data_fs', data=images, compression="gzip")  # 使用 gzip 压缩数据

	print(f"Images saved to {output_mat_file} (HDF5 format)")



# 使用示例
image_folder = r'D:\syndiff\datasets\test\CT'  # 请替换为图像文件夹的路径
output_hdf5_file = './dataset/test_CT.mat'  # 目标 HDF5 文件路径
save_images_to_hdf5(image_folder, output_hdf5_file)

with h5py.File('./dataset/train_MR.mat', 'r') as f:
	print(list(f.keys()))