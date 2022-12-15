import json
import numpy as np
import torch
from utils import show
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader

class NerfDataLoader(Dataset):
	def __init__(self, jsonPath, imageWidth=128, imageHeight=128, datasetPath='', dataType='train'):

		super(NerfDataLoader, self).__init__()

		# Reading the JSON data
		with open(jsonPath, 'r') as file:
		    jsonData = json.load(file)

		# Parsing the JSON file
		self.imagesPath = []
		self.c2wMatrix  = [] # Camera2World matrix
		for frame in jsonData['frames']:
			imagePath = frame['file_path'].replace('.', datasetPath)
			self.imagesPath.append('{}.png'.format(imagePath))
			self.c2wMatrix.append(frame['transform_matrix'])

		self.imageHeight = imageHeight
		self.imageWidth  = imageWidth

	def __len__(self):
		return len(self.imagesPath)

	def __getitem__(self, idx):
		image = io.read_image(self.imagesPath[idx], mode=io.ImageReadMode.RGB).to(torch.float32)/255.0
		image = transforms.Resize((self.imageHeight, self.imageWidth))(image)
		c2w   = torch.FloatTensor(self.c2wMatrix[idx])
		return (image, c2w)

if __name__ == '__main__':
	image_height = 128
	image_width  = 128
	batch_size   = 8
	dataloader = DataLoader(NerfDataLoader(jsonPath='dataset/nerf_synthetic/ship/transforms_train.json', datasetPath='dataset/nerf_synthetic/ship', imageHeight=image_height, imageWidth=image_width), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
	image, c2wMatrix = next(iter(dataloader))
	print(image.shape, c2wMatrix.shape)
	show(image.to(torch.uint8))