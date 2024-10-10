
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import os


class WM38Dataset(Dataset):
	def __init__(self, data_x, data_y, transform=None):
		self.transform = transform
		self.images = data_x
		self.labels = data_y

	def __len__(self):
		# 返回数据集的数据数量
		return len(self.images)
 
	def __getitem__(self, index):
		img = self.images[index]
		label = self.labels[index]
		img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
		img = np.where(img == 1, 128, img)
		img = np.where(img == 2, 255, img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		
		sample = {'data': img, 'label': label}
		if self.transform:
			sample['data'] = self.transform(sample['data'])
		return sample
