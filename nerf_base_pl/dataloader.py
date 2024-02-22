import os
import cv2
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
import lightning as L
from utils import show
from natsort import natsorted
from torchvision import transforms, io
from data_module.real_llff_data import load_llff_data
from torch.utils.data import Dataset, DataLoader


class NerfDataLoader(Dataset):
	def __init__(self, args, stage='train'):

		super(NerfDataLoader, self).__init__()

		if (args.dataset_type == 'real') or (args.dataset_type == 'llff'):

			if stage == 'train':
				images, poses, near_far, _, _ = load_llff_data(basedir=args.basedir, factor=args.scale, recenter=True, bd_factor=.75, spherify=args.spherify)
			elif stage == 'val':
				images, poses, near_far, _, _ = load_llff_data(basedir=args.basedir, factor=args.scale, recenter=True, bd_factor=.75, spherify=args.spherify)
			elif stage == 'test':
				pass

			# Reading the numpy file
			if stage == 'train':
				camera_path = os.path.join(args.basedir, 'poses_bounds.npy')
				np_data = np.load(camera_path)
			elif stage == 'val':
				camera_path = os.path.join(args.basedir, 'poses_bounds.npy')
				np_data = np.load(camera_path)
			elif stage == 'test':
				pass

			# Parsing the numpy file
			# near_far     = bds
			# camera_param = np_data[..., :-2].reshape(-1, 3, 5) # 3 x 5
			hwf          = poses[..., -1] # Height, Width, Focal
			camera_mat   = poses[..., :-1] # 3 x 4

			# Parsing the JSON file
			self.images_path = []
			self.c2wMatrix   = [] # Camera2World matrix
			self.focal       = []
			self.direction   = []
			self.bounds      = []
			self.image_width  = args.image_width // args.scale
			self.image_height = args.image_height // args.scale

			# Pixel of image
			x, y = np.meshgrid(
									np.arange(0, self.image_width, dtype=np.float32),
									np.arange(0, self.image_height, dtype=np.float32),
									indexing = 'xy'
								)

			for idx in tqdm(range(images.shape[0])):
				# Calculating downscale factor
				focal_len = hwf[idx, 2]

				# Pixel to camera coordinates
				camera_x = (x - (self.image_width  * 0.5))/focal_len
				camera_y = (y - (self.image_height * 0.5))/focal_len

				# creating a direction vector and normalizing to unit vector
				# direction of pixels w.r.t local camera origin (0,0,0)
				# dirc = np.stack([camera_x, -camera_y, np.ones_like(camera_x)], axis=-1)
				# Thanks to this person :) [https://github.com/sillsill777]
				dirc = np.stack([camera_x, -camera_y, -np.ones_like(camera_x)], axis=-1)
				self.direction.append(dirc)
				images[idx] = np.uint8(np.clip(images[idx]*255.0, 0, 255))
				self.images_path.append(np.float32(cv2.resize(images[idx], (self.image_width, self.image_height)))/255.0)
				self.c2wMatrix.append(camera_mat[idx])
				self.focal.append(focal_len)
				self.bounds.append(near_far[idx])
				# self.bounds.append([0., 1.])

		elif args.dataset_type=='synthetic':

			if stage == 'train':
				# Reading the JSON data
				camera_path = os.path.join(args.basedir, 'transforms_train.json')
				with open(camera_path, 'r') as file:
					json_data = json.load(file)
			elif stage == 'val':
				camera_path = os.path.join(args.basedir, 'transforms_val.json')
				with open(camera_path, 'r') as file:
					json_data = json.load(file)
			elif stage == 'test':
				pass

			# Parsing the JSON file

			self.images_path = []
			self.c2wMatrix   = [] # Camera2World matrix
			self.focal       = []
			self.direction   = []
			self.bounds      = []
			self.image_width  = args.image_width // args.scale
			self.image_height = args.image_height // args.scale

			for frame in tqdm(json_data['frames']):
				focal_len = ((args.image_width/2.0)/(math.tan(json_data['camera_angle_x']/2.0)))/args.scale # f =  (p/2) / tan(theta/2)

				# Pixel of image
				x, y = np.meshgrid(
										np.arange(0, self.image_width, dtype=np.float32),
										np.arange(0, self.image_height, dtype=np.float32),
										indexing = 'xy'
									)

				# Pixel to camera coordinates
				camera_x = (x - (self.image_width  * 0.5))/focal_len
				camera_y = (y - (self.image_height * 0.5))/focal_len

				# creating a direction vector and normalizing to unit vector
				# direction of pixels w.r.t local camera origin (0,0,0)
				dirc = np.stack([camera_x, -camera_y, -np.ones_like(camera_x)], axis=-1)
				self.direction.append(dirc)
				self.focal.append(focal_len)
				imagePath = frame['file_path'].replace('.', args.basedir)
				image     = io.read_image('{}.png'.format(imagePath), mode=io.ImageReadMode.RGB).to(torch.float32)/255.0
				image     = transforms.Resize((self.image_height, self.image_width))(image)
				image     = torch.permute(image, (1, 2, 0))
				self.images_path.append(image)
				self.c2wMatrix.append(frame['transform_matrix'])
				self.bounds.append([args.near_plane, args.far_plane])


	def __len__(self):
		return len(self.images_path)

	def __getitem__(self, idx):
		image     = torch.FloatTensor(self.images_path[idx])
		c2w       = torch.FloatTensor(self.c2wMatrix[idx])
		focal     = torch.as_tensor(self.focal[idx], dtype=torch.float32) 
		direction = torch.FloatTensor(self.direction[idx])
		near      = torch.as_tensor(self.bounds[idx][0], dtype=torch.float32) 
		far       = torch.as_tensor(self.bounds[idx][1], dtype=torch.float32) 
		return (image, c2w, focal, direction, near, far)

class NerfDataLoaderLight(L.LightningDataModule):
	def __init__(self, args):
		super(NerfDataLoaderLight, self).__init__()
		self.args = args
	
	def setup(self, stage):
		if stage == 'fit':
			self.train_data = NerfDataLoader(args = self.args, stage='train')
			self.val_data   = NerfDataLoader(args = self.args, stage='val')
		elif stage == 'test':
			self.test_data   = NerfDataLoader(args = self.args, stage='test')
		elif stage == 'predict':
			pass
	
	def train_dataloader(self):
		return DataLoader(self.train_data,\
						  batch_size=self.args.batch_size,\
						  shuffle=True,\
						  num_workers=self.args.workers,\
						  pin_memory=self.args.pin_memory,\
						  drop_last=False)
	
	def val_dataloader(self):
		return DataLoader(self.val_data,\
						  batch_size=self.args.batch_size,\
						  shuffle=False,\
						  num_workers=self.args.workers,\
						  pin_memory=self.args.pin_memory,\
						  drop_last=False)
	
	def test_dataloader(self):
		pass
		# return DataLoader(self.val_data,\
		# 				  batch_size=self.args.batch_size,\
		# 				  shuffle=True,\
		# 				  num_workers=self.args.workers,\
		# 				  pin_memory=self.args.pin_memory,\
		# 				  drop_last=False)
	
	def predict_dataloader(self):
		pass
		# return DataLoader(self.val_data,\
		# 				  batch_size=self.args.batch_size,\
		# 				  shuffle=True,\
		# 				  num_workers=self.args.workers,\
		# 				  pin_memory=self.args.pin_memory,\
		# 				  drop_last=False)