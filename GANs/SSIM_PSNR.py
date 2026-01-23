from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pytorch_fid.fid_score import calculate_fid_given_paths
import cv2
import os
import torch.fft as fft
import torch
import cv2
import math
import lpips
import itertools
from tqdm import tqdm

def decompose(img):
	# Load an image
	# img = cv2.imread('H:/BraTS2021_00000_flair_60.png')
	img = img

	# Convert the image to a tensor
	img_tensor = torch.from_numpy(img).permute(2, 0, 1)
	img_tensor = img_tensor

	# Perform the Fourier transform
	fourier_transform = fft.rfft2(img_tensor)
	fourier_transform = fft.fftshift(fourier_transform)

	C, W, H = fourier_transform.shape[0], fourier_transform.shape[1], fourier_transform.shape[2]
	mask1 = torch.zeros(C, W, H)
	mask2 = torch.zeros(C, W, H)
	D = 20;
	D2 = 5;
	c_w = W / 2;
	c_h = H / 2

	# 高斯滤波器
	distance1 = torch.zeros(C, W, H)

	for w in range(W):
		for h in range(H):
			distance1[:, w, h] = math.sqrt((pow((w - c_w), 2)) + pow((h - c_h), 2))
			mask1[:, w, h] = 1 - torch.exp(-(torch.pow(distance1[:, w, h], 2)) / (2 * D * D))  # 高通
			mask2[:, w, h] = torch.exp(-(torch.pow(distance1[:, w, h], 2)) / (2 * D * D))  # 低通

	# 滤波
	fourier_transform_1 = mask1 * fourier_transform
	fourier_transform_2 = mask2 * fourier_transform

	fourier_transform_1 = fft.ifftshift(fourier_transform_1)
	fourier_transform_2 = fft.ifftshift(fourier_transform_2)

	# Perform the inverse Fourier transform
	filtered_img_tensor_1 = fft.irfft2(fourier_transform_1)
	filtered_img_tensor_2 = fft.irfft2(fourier_transform_2)
	# print(torch.max(filtered_img_tensor_2))
	# save_image(filtered_img_tensor_2/255.0, './Lowfreq.png') # save_image 要把图像归一化到0~1，即除以255.0

	# save images
	# filtered_img_tensor_1 = 255 * (filtered_img_tensor_1 * 0.5)
	# filtered_img_tensor_2 = 255 * (filtered_img_tensor_2 * 0.5)

	# Convert the result back to an image
	filtered_img_1 = (filtered_img_tensor_1.permute(1, 2, 0)/255).numpy()
	filtered_img_2 = (filtered_img_tensor_2.permute(1, 2, 0)/255).numpy()

	return filtered_img_1, filtered_img_2

def FID_score(path1, path2):
	fid_value = calculate_fid_given_paths(
		[path1, path2],
		batch_size=10,  # adjust based on your GPU memory
		device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		dims=2048  # default InceptionV3 feature dimension
	)
	print(fid_value,"\n")


def Evaluation_function(mode, n=480):
	truth_path = 'D:/CT2MR/CycleGAN-master/PyTorch-CycleGAN-master/baseline/CycleGAN/PyTorch-CycleGAN-master/datasets/BraTS2021/test/'+ mode[0]
	generated_path = 'D:/CT2MR/CycleGAN-master/PyTorch-CycleGAN-master/baseline/CycleGAN/PyTorch-CycleGAN-master/logs/'+ mode[1]

	p=0; m=0; s=0
	p_L=0; m_L=0; s_L=0

	for roots, dirs, files in os.walk(truth_path):
		for file in files:
			img1 = cv2.imread(os.path.join(roots, file))
			img2 = cv2.imread(os.path.join(generated_path, file))

			p = p + compare_psnr(img1, img2)
			s = s + compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
			m = m + compare_mse(img1, img2)

	# for roots, dirs, files in os.walk(truth_path):
	# 	for file in files:
	# 		img1_H, img1_L = decompose(cv2.imread(os.path.join(roots, file)))
	# 		img2_H, img2_L = decompose(cv2.imread(os.path.join(generated_path, file)))
	#
	# 		p = p + compare_psnr(img1_H, img2_H)
	# 		p_L = p_L + compare_psnr(img1_L, img2_L)
	#
	# 		s = s + compare_ssim(img1_H, img2_H, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
	# 		s_L = s_L + compare_ssim(img1_L, img2_L, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
	#
	# 		m = m + compare_mse(img1_H, img2_H)
	# 		m_L = m_L + compare_mse(img1_L, img2_L)


	# print(mode[0] + ':', 'PSNR：{}，SSIM：{}，MSE：{}'.format(p/960, s/960, m/960))
	print(mode[0] + ':', 'PSNR：{}，SSIM：{}，MSE：{}'.format(p / n, s / n, m / n))


def calculate_average_lpips(path1, path2):
	"""
	计算两个目录中所有图像的平均LPIPS分数（非配对）

	Args:
		path1 (str): 第一个图像目录
		path2 (str): 第二个图像目录

	Returns:
		float: 平均LPIPS分数
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# 初始化LPIPS损失函数
	loss_fn = lpips.LPIPS(net='alex', version='0.1')
	loss_fn = loss_fn.to(device)

	# 获取两个目录中的所有图像
	images1 = [os.path.join(path1, f) for f in os.listdir(path1)
	           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	images2 = [os.path.join(path2, f) for f in os.listdir(path2)
	           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

	lpips_scores = []

	# 计算所有图像对的LPIPS分数
	for img1_path, img2_path in tqdm(list(itertools.product(images1, images2)),
	                                 desc="计算LPIPS分数"):
		try:
			# 加载并预处理图像
			img1 = lpips.load_image(img1_path)
			img2 = lpips.load_image(img2_path)

			# 转换为tensor
			img1 = lpips.im2tensor(img1).to(device)
			img2 = lpips.im2tensor(img2).to(device)

			# 计算LPIPS分数
			with torch.no_grad():
				lpips_score = loss_fn(img1, img2)
				lpips_scores.append(lpips_score.item())

		except Exception as e:
			print(f"处理图像对 {img1_path} 和 {img2_path} 时出错: {e}")

	# 计算平均LPIPS分数
	if lpips_scores:
		print(sum(lpips_scores) / len(lpips_scores), "\n")
	else:
		return None

if __name__ == '__main__':
	Evaluation_function(['Flair', 'B'])
	Evaluation_function(['T2', 'A'])
	Evaluation_function(['T1ce', 'B'])
	Evaluation_function(['T1', 'A'])
	Evaluation_function(['T1_sub', 'deblur'])
	Evaluation_function(['T1_sub', 'denoised'])

	# print("to CT:")
	# FID_score('./logs/CT/', './datasets/CHAOS/test/CT/')
	# print("to MR:")
	# FID_score('./logs/MR/', './datasets/CHAOS/test/MR/')
	# print("to CT:")
	# calculate_average_lpips('./logs/CT/', './datasets/CHAOS/test/CT/')
	# print("to MR:")
	# calculate_average_lpips('./logs/MR/', './datasets/CHAOS/test/MR/')


