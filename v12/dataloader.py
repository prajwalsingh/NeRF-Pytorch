import json
import numpy as np
import torch
import math
import config
from glob import glob
from natsort import natsorted
from utils import show
from torchvision import transforms, io
from torch.utils.data import Dataset, DataLoader

class NerfDataLoader(Dataset):
	def __init__(self, camera_path, imageWidth=128, imageHeight=128, data_path='', dataType='train', pre_height=800, pre_width=800):

		super(NerfDataLoader, self).__init__()

		self.imageHeight = imageHeight
		self.imageWidth  = imageWidth

		if (config.dataset_type == 'real') or (config.dataset_type == 'llff'):

			# Reading the numpy file
			np_data = np.load(camera_path)

			# Parsing the numpy file
			near_far     = np_data[..., -2:]
			camera_param = np_data[..., :-2].reshape(-1, 3, 5) # 3 x 5
			hwf          = camera_param[..., camera_param.shape[1]-1] # Height, Width, Focal
			camera_mat   = camera_param[..., :-1] # 3 x 4

			# Parsing the JSON file
			self.images_path = []
			self.c2wMatrix  = [] # Camera2World matrix
			self.focal      = []
			self.direction  = []
			self.bounds     = []

			for idx, frame in enumerate(natsorted(glob(data_path+'/*'))):
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
				self.direction.append(dirc)
				
				self.images_path.append(frame)
				self.c2wMatrix.append(camera_mat[idx])
				self.focal.append(focal_len)
				self.bounds.append(near_far[idx])

		elif config.dataset_type=='json':
			# Reading the JSON data
			with open(camera_path, 'r') as file:
				json_data = json.load(file)

			# Parsing the JSON file

			self.images_path = []
			self.c2wMatrix   = [] # Camera2World matrix
			self.focal       = []
			self.direction   = []
			self.bounds      = []

			for frame in json_data['frames']:
				downscale = pre_width / self.imageWidth
				focal_len = ((pre_width/2.0)/(math.tan(json_data['camera_angle_x']/2.0)))/downscale # f =  (p/2) / tan(theta/2)

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
				dirc = np.stack([camera_x, -camera_y, -np.ones_like(camera_x)], axis=-1)
				self.direction.append(dirc)
				self.focal.append(focal_len)
				imagePath = frame['file_path'].replace('.', data_path)
				self.images_path.append('{}.png'.format(imagePath))
				self.c2wMatrix.append(frame['transform_matrix'])
				self.bounds.append([config.near_plane, config.far_plane])


	def __len__(self):
		return len(self.images_path)

	def __getitem__(self, idx):
		image     = io.read_image(self.images_path[idx], mode=io.ImageReadMode.RGB).to(torch.float32)/255.0
		image     = transforms.Resize((self.imageHeight, self.imageWidth))(image)
		c2w       = torch.FloatTensor(self.c2wMatrix[idx])
		focal     = torch.as_tensor(self.focal[idx], dtype=torch.float32) 
		direction = torch.FloatTensor(self.direction[idx])
		near      = torch.as_tensor(self.bounds[idx][0], dtype=torch.float32) 
		far       = torch.as_tensor(self.bounds[idx][1], dtype=torch.float32) 
		
		return (image, c2w, focal, direction, near, far)

if __name__ == '__main__':
	image_height = 128
	image_width  = 128
	batch_size   = 8
	dataloader = DataLoader(NerfDataLoader(jsonPath='dataset/nerf_synthetic/ship/transforms_train.json', datasetPath='dataset/nerf_synthetic/ship', imageHeight=image_height, imageWidth=image_width), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
	image, c2wMatrix = next(iter(dataloader))
	print(image.shape, c2wMatrix.shape)
	show(image.to(torch.uint8))