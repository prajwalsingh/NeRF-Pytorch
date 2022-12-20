import json
import numpy as np
import torch
import math
import config
from glob import glob
from natsort import natsorted
from utils import show_test
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader

class NerfDataLoader360(Dataset):
	def __init__(self, jsonPath, imageWidth=128, imageHeight=128, datasetPath='', dataType='train'):

		super(NerfDataLoader360, self).__init__()

		self.imageHeight = imageHeight
		self.imageWidth  = imageWidth

		# Reading the numpy file
		np_data = np.load(jsonPath)

		# Parsing the numpy file
		near_far     = np_data[..., -2:]
		camera_param = np_data[..., :-2].reshape(-1, 3, 5)
		hwf          = camera_param[..., camera_param.shape[1]-1] # Height, Width, Focal
		camera_mat   = camera_param[..., :-1]

		# Parsing the JSON file
		self.imagesPath = []
		self.c2wMatrix  = [] # Camera2World matrix
		self.focal      = []
		self.direction  = []
		self.bounds     = []

		for idx, frame in enumerate(natsorted(glob(datasetPath+'/*'))):
			# Calculating downscale factor
			downscale = hwf[idx, 1] / self.imageWidth
			focal_len = hwf[idx, 2] /downscale

			# Pixel of image
			x, y = np.meshgrid(
									np.arange(0, self.imageWidth, dtype=np.float32),
									np.arange(0, self.imageHeight, dtype=np.float32),
									indexing = 'xy'
								 )

			# Pixel to camera coordinates
			camera_x = (x - (self.imageWidth  * 0.5))/focal_len
			camera_y = (y - (self.imageHeight * 0.5))/focal_len

			# creating a direction vector and normalizing to unit vector
			# direction of pixels w.r.t local camera origin (0,0,0)
			dirc = np.stack([camera_x, -camera_y, np.ones_like(camera_x)], axis=-1)
			dirc = dirc / np.expand_dims(np.linalg.norm(x=dirc, axis=-1), axis=-1)
			self.direction.append(dirc)
			
			self.imagesPath.append(frame)
			self.c2wMatrix.append(camera_mat[idx])
			self.focal.append(focal_len)
			self.bounds.append(near_far[idx])

	def __len__(self):
		return len(self.imagesPath)

	def __getitem__(self, idx):
		image = io.read_image(self.imagesPath[idx], mode=io.ImageReadMode.RGB).to(torch.float32)/255.0
		H, W, C   = image.shape
		image     = transforms.Resize((self.imageHeight, self.imageWidth))(image)
		c2w       = torch.FloatTensor(self.c2wMatrix[idx])
		focal     = torch.as_tensor(self.focal[idx], dtype=torch.float32) 
		direction = torch.FloatTensor(self.direction[idx])
		near      = torch.as_tensor(self.bounds[idx][0], dtype=torch.float32) 
		far       = torch.as_tensor(self.bounds[idx][1], dtype=torch.float32) 
		return (image, c2w, focal, direction, near, far)

if __name__ == '__main__':
	dataloader = DataLoader(NerfDataLoader360(jsonPath=config.train_json_path,\
										   datasetPath=config.image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size,\
										   shuffle=True, num_workers=8,\
										   pin_memory=True, drop_last=True)
	image, c2wMatrix, focal, direction = next(iter(dataloader))
	print(image.shape, c2wMatrix.shape, focal.shape, direction.shape)
	show_test(image)