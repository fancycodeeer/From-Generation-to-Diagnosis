import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train', data='CT2MRI', get_name=True):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.get_name = get_name

        if data == 'CT2MRI':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/CT' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/MRI' % mode) + '/*.*'))
        elif data == 'BraTS2021':
            # self.files_A = sorted(glob.glob(os.path.join(root, '%s/T2' % mode) + '/*.*'))
            # self.files_B = sorted(glob.glob(os.path.join(root, '%s/Flair' % mode) + '/*.*'))
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/T1' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/T1ce' % mode) + '/*.*'))
        elif data == 'IXI':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/T2' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/PD' % mode) + '/*.*'))
        elif data == 'horse2zebra':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/horse' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/zebra' % mode) + '/*.*'))
        elif data == 'CHAOS':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/CT' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/MR' % mode) + '/*.*'))
        elif data == 'Carla':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/day' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/night' % mode) + '/*.*'))
        elif data == 'Denoise':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/T1_noisy' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/T1_sub' % mode) + '/*.*'))
        elif data == 'Deblur':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/T1_blur' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/T1_sub' % mode) + '/*.*'))
        elif data is None:
            self.files_A = sorted(glob.glob(os.path.join(root, 'T2') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, 'Flair') + '/*.*'))


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        if self.get_name:
            file_name_A = self.files_A[index % len(self.files_A)].split("\\")[-1]

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
            if self.get_name:
                file_name_B = self.files_B[random.randint(0, len(self.files_B) - 1)].split("\\")[-1]

        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
            if self.get_name:
                file_name_B = self.files_B[index % len(self.files_B)].split("\\")[-1]

        if self.get_name:
            return {'A': item_A, 'B': item_B, 'name_A': file_name_A, 'name_B': file_name_B}
        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



