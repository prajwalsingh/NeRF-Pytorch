# https://keras.io/examples/vision/nerf/
import os
import json
import math
import shutil
import random
import numpy as np
import torch
import config
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from nerf_model import NerfNet
from nerf_components import NerfComponents
from piq import psnr

import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from dataloader import NerfDataLoader
from utils import show
from glob import glob
from natsort import natsorted

torch.autograd.set_detect_anomaly(True)
plt.rcParams["savefig.bbox"] = 'tight'
style.use('seaborn')
torch.manual_seed(45)
np.random.seed(45)
random.seed(45)


if __name__ == '__main__':
	
	#########################################################################################
	train_dataloader = DataLoader(NerfDataLoader(jsonPath=config.train_json_path,\
										   datasetPath=config.image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size, shuffle=True,\
										   num_workers=8, pin_memory=True, drop_last=True)

	val_dataloader = DataLoader(NerfDataLoader(jsonPath=config.val_json_path,\
										   datasetPath=config.image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size, shuffle=True,\
										   num_workers=8, pin_memory=True, drop_last=True)
	base_image, base_c2wMatrix, base_focal = next(iter(val_dataloader))
	base_image, base_c2wMatrix = base_image.to(config.device),\
								 base_c2wMatrix.to(config.device)
	base_focal = base_focal[:1]
	# print(image.shape, c2wMatrix.shape)
	# show(image.to(torch.uint8))
	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   focal=base_focal.to(torch.float32).to(config.device),\
							   batch_size=config.batch_size,\
							   near=config.near_plane,\
							   far=config.far_plane,\
							   num_samples=config.num_samples,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	nerfnet_coarse = NerfNet(depth=config.net_depth, xyz_feat=config.xyz_feat,\
					  		 dirc_feat=config.dirc_feat, net_dim=config.net_dim,\
					  		 skip_layer=config.skip_layer).to(config.device)

	nerfnet_fine   = NerfNet(depth=config.net_depth, xyz_feat=config.xyz_feat,\
					  		 dirc_feat=config.dirc_feat, net_dim=config.net_dim,\
					  		 skip_layer=config.skip_layer).to(config.device)

	nerfnet_coarse = torch.nn.DataParallel(nerfnet_coarse).to(config.device)
	nerfnet_fine   = torch.nn.DataParallel(nerfnet_fine).to(config.device)

	# optimizer_coarse = torch.optim.Adam(nerfnet_coarse.parameters(), lr=config.lr)
	# optimizer_fine   = torch.optim.Adam(nerfnet_fine.parameters(), lr=config.lr)

	# loss_fn_coarse   = torch.nn.MSELoss()
	# loss_fn_fine     = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(\
									list(nerfnet_coarse.parameters()) +\
									list(nerfnet_fine.parameters()),
									lr=config.lr
								)
	loss_fn = torch.nn.MSELoss()

	START_EPOCH = 1
	END_EPOCH   = config.epochs
	#########################################################################################

	#########################################################################################
	dir_info  = natsorted(glob('EXPERIMENT_*'))

	if len(dir_info)==0:
		experiment_num = 1
	else:
		experiment_num = int(dir_info[-1].split('_')[-1]) + 1

	if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
		os.makedirs('EXPERIMENT_{}'.format(experiment_num))
		os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))

	ckpt_path = 'EXPERIMENT_{}/checkpoints/nerf_1.pth'.format(experiment_num)

	if os.path.isfile(ckpt_path):
		ckpt_path  = natsorted(glob('EXPERIMENT_{}/checkpoints/nerf_*.pth'.format(experiment_num)))[-1]
		checkpoint = torch.load(ckpt_path)
		nerfnet_coarse.load_state_dict(checkpoint['model_state_dict_coarse'])
		# optimizer_coarse.load_state_dict(checkpoint['optimizer_state_dict_coarse'])
		nerfnet_fine.load_state_dict(checkpoint['model_state_dict_fine'])
		# optimizer_fine.load_state_dict(checkpoint['optimizer_state_dict_fine'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		START_EPOCH = checkpoint['epoch']
		print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
		START_EPOCH += 1
	else:
		os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
	#########################################################################################

	#########################################################################################
	for epoch in range(START_EPOCH, END_EPOCH):

		nerfnet_coarse.train()
		nerfnet_fine.train()

		train_loss_tracker = [0.0]
		train_psnr_tracker = [0.0]
		tq = tqdm(train_dataloader)
		# print(list(nerfnet.parameters())[0])

		for idx, (image, c2wMatrix, _) in enumerate(tq, start=1):

			temp_loss_tracker = [0.0]
			total_loss        = 0.0

			image, c2wMatrix  = image.to(config.device), c2wMatrix.to(config.device)
			image             = torch.permute(image, (0, 2, 3, 1))
			# rays, t_vals     = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)
			xyz, dirc, t_vals = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)

			image  = image.reshape(config.batch_size, -1, 3)

			# print(xyz.shape, dirc.shape, t_vals.shape)
			# rays, t_vals, image  = nerf_comp.sub_batching(rays, t_vals, image, chunk_size=config.chunk_size)
			xyz, dirc, t_vals, image  = nerf_comp.sub_batching(xyz, dirc, t_vals, image, chunk_size=config.chunk_size)

			# for idx, (ray_chunk, t_val_chunk, image_chunk) in enumerate(zip(rays, t_vals, image)):
			for idx, (xyz_chunk, dirc_chunk, t_val_chunk, image_chunk) in enumerate(zip(xyz, dirc, t_vals, image)):

				# print(xyz_chunk.shape, dirc_chunk.shape, t_val_chunk.shape, image_chunk.shape)
				optimizer.zero_grad()

				# prediction   = nerfnet_coarse(xyz, dirc)
				density, rgb   = nerfnet_coarse(xyz_chunk, dirc_chunk)

				# print(density.shape, rgb.shape)
				# prediction   = torch.reshape(prediction,\
				# 							(config.batch_size,\
				# 							config.image_height//config.chunk_size,\
				# 							config.image_width,\
				# 							config.num_samples,\
				# 							prediction.shape[-1]))

				rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_chunk, dirc=dirc_chunk, t_vals=t_val_chunk, random_sampling=True)

				# fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)
				xyz_fine, dirc_fine, t_val_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)

				density, rgb   = nerfnet_fine(xyz_fine, dirc_fine)

				# prediction   = torch.reshape(prediction,\
				# 							(config.batch_size,\
				# 							config.image_height,\
				# 							config.image_width,\
				# 							config.num_samples_fine,\
				# 							prediction.shape[-1]))

				rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_fine, dirc=dirc_fine, t_vals=t_val_fine, random_sampling=True)

				loss = loss_fn(image_chunk, rgb_coarse) + loss_fn(image_chunk, rgb_fine)
				# total_loss = total_loss + loss
				loss.backward()

				temp_loss_tracker.append(loss.detach().cpu())
				# train_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())
				train_loss_tracker.append(sum(temp_loss_tracker)/len(temp_loss_tracker))

				# tq.set_description('E: {}, tr_loss: {:0.3f}'.format(epoch, loss_tracker/(idx*config.batch_size)))

				del rgb_coarse, depth_map_coarse, weights_coarse, xyz_chunk, dirc_chunk, t_val_chunk, xyz_fine, dirc_fine, t_val_fine, rgb_fine, depth_map_fine, weights_fine, loss, density, rgb
				# break

			optimizer.step()
			# tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(train_loss_tracker)/len(train_loss_tracker), sum(train_psnr_tracker)/len(train_psnr_tracker)))
			tq.set_description('E: {}, TL: {:0.3f}'.format(epoch, sum(train_loss_tracker)/len(train_loss_tracker)))
			# break

		with torch.no_grad():
			val_loss_tracker = [0.0]
			val_psnr_tracker = [0.0]
			tq = tqdm(val_dataloader)
			# print(list(nerfnet.parameters())[0])

			for idx, (image, c2wMatrix, _) in enumerate(tq, start=1):

				temp_loss_tracker = [0.0]

				image, c2wMatrix = image.to(config.device), c2wMatrix.to(config.device)
				image            = torch.permute(image, (0, 2, 3, 1))
				# rays, t_vals     = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)
				xyz, dirc, t_vals = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)

				image  = image.reshape(config.batch_size, -1, 3)

				# print(xyz.shape, dirc.shape, t_vals.shape)
				# rays, t_vals, image  = nerf_comp.sub_batching(rays, t_vals, image, chunk_size=config.chunk_size)
				xyz, dirc, t_vals, image  = nerf_comp.sub_batching(xyz, dirc, t_vals, image, chunk_size=config.chunk_size)

				# for idx, (ray_chunk, t_val_chunk, image_chunk) in enumerate(zip(rays, t_vals, image)):
				for idx, (xyz_chunk, dirc_chunk, t_val_chunk, image_chunk) in enumerate(zip(xyz, dirc, t_vals, image)):

					# print(xyz_chunk.shape, dirc_chunk.shape, t_val_chunk.shape, image_chunk.shape)
					# prediction   = nerfnet_coarse(xyz, dirc)
					density, rgb   = nerfnet_coarse(xyz_chunk, dirc_chunk)

					# print(density.shape, rgb.shape)
					# prediction   = torch.reshape(prediction,\
					# 							(config.batch_size,\
					# 							config.image_height//config.chunk_size,\
					# 							config.image_width,\
					# 							config.num_samples,\
					# 							prediction.shape[-1]))

					rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_chunk, dirc=dirc_chunk, t_vals=t_val_chunk, random_sampling=True)

					# fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)
					xyz_fine, dirc_fine, t_val_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)

					density, rgb   = nerfnet_fine(xyz_fine, dirc_fine)

					# prediction   = torch.reshape(prediction,\
					# 							(config.batch_size,\
					# 							config.image_height,\
					# 							config.image_width,\
					# 							config.num_samples_fine,\
					# 							prediction.shape[-1]))

					rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_fine, dirc=dirc_fine, t_vals=t_val_fine, random_sampling=True)

					loss = loss_fn(image_chunk, rgb_coarse) + loss_fn(image_chunk, rgb_fine)

					temp_loss_tracker.append(loss.detach().cpu())
					# train_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())
					val_loss_tracker.append(sum(temp_loss_tracker)/len(temp_loss_tracker))

					# tq.set_description('E: {}, tr_loss: {:0.3f}'.format(epoch, loss_tracker/(idx*config.batch_size)))

					del rgb_coarse, depth_map_coarse, weights_coarse, xyz_chunk, dirc_chunk, t_val_chunk, xyz_fine, dirc_fine, t_val_fine, rgb_fine, depth_map_fine, weights_fine, loss, density, rgb
					# break

				# tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
				tq.set_description('E: {}, TL: {:0.3f}'.format(epoch, sum(val_loss_tracker)/len(val_loss_tracker)))
				# break

			rgb_final, depth_final = [], []
			xyz, dirc, t_vals = nerf_comp.sampling_rays(camera_matrix=base_c2wMatrix, random_sampling=True)
			image  = torch.permute(base_image, (0, 2, 3, 1))
			image  = image.reshape(config.batch_size, -1, 3)
			# print(xyz.shape, dirc.shape, t_vals.shape)
			# rays, t_vals, image  = nerf_comp.sub_batching(rays, t_vals, image, chunk_size=config.chunk_size)
			xyz, dirc, t_vals, image  = nerf_comp.sub_batching(xyz, dirc, t_vals, image, chunk_size=config.chunk_size)

			# for idx, (ray_chunk, t_val_chunk, image_chunk) in enumerate(zip(rays, t_vals, image)):
			for idx, (xyz_chunk, dirc_chunk, t_val_chunk, image_chunk) in enumerate(zip(xyz, dirc, t_vals, image)):

				# print(xyz_chunk.shape, dirc_chunk.shape, t_val_chunk.shape, image_chunk.shape)
				# prediction   = nerfnet_coarse(xyz, dirc)
				density, rgb   = nerfnet_coarse(xyz_chunk, dirc_chunk)

				# print(density.shape, rgb.shape)
				# prediction   = torch.reshape(prediction,\
				# 							(config.batch_size,\
				# 							config.image_height//config.chunk_size,\
				# 							config.image_width,\
				# 							config.num_samples,\
				# 							prediction.shape[-1]))

				rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_chunk, dirc=dirc_chunk, t_vals=t_val_chunk, random_sampling=True)

				# fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)
				xyz_fine, dirc_fine, t_val_fine = nerf_comp.sampling_fine_rays(camera_matrix=base_c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)

				density, rgb   = nerfnet_fine(xyz_fine, dirc_fine)

				# prediction   = torch.reshape(prediction,\
				# 							(config.batch_size,\
				# 							config.image_height,\
				# 							config.image_width,\
				# 							config.num_samples_fine,\
				# 							prediction.shape[-1]))

				rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(density=density, rgb=rgb, xyz=xyz_fine, dirc=dirc_fine, t_vals=t_val_fine, random_sampling=True)
				rgb_final.append(rgb_fine)
				depth_final.append(depth_map_fine)

			rgb_final = torch.concat(rgb_final, dim=-2).reshape(config.batch_size, config.image_height, config.image_width, -1)
			rgb_final = torch.permute(rgb_final, (0, 3, 1, 2))
			depth_final = torch.concat(depth_final, dim=-2).reshape(config.batch_size, config.image_height, config.image_width)

			show(imgs=rgb_final[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
			show(imgs=depth_final[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)

			del rgb_final, depth_final

		torch.save({
					'epoch': epoch,
					'model_state_dict_coarse': nerfnet_coarse.state_dict(),
					# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
					'model_state_dict_fine': nerfnet_fine.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
					# 'scheduler_state_dict': scheduler.state_dict()
			}, 'EXPERIMENT_{}/checkpoints/nerf_{}.pth'.format(experiment_num, epoch))

		with open('EXPERIMENT_{}/log.txt'.format(experiment_num), 'a') as file:
			file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}, VL: {:0.3f}, VPSNR: {:0.3f}\n'.\
				format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
				sum(train_psnr_tracker)/len(train_psnr_tracker),\
				sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
		# #########################################################################################