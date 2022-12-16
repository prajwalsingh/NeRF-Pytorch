# https://keras.io/examples/vision/nerf/
import os
import json
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
	base_image, base_c2wMatrix = next(iter(val_dataloader))
	# print(image.shape, c2wMatrix.shape)
	# show(image.to(torch.uint8))
	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   focal=config.focal,\
							   batch_size=config.batch_size,\
							   near=config.near_plane,\
							   far=config.far_plane,\
							   num_samples=config.num_samples,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	nerfnet_coarse = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_fine   = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

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

	ckpt_path = 'EXPERIMENT_{}/nerf_base.pth'.format(experiment_num)

	if os.path.isfile(ckpt_path):		
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
	#########################################################################################

	#########################################################################################
	for epoch in range(START_EPOCH, END_EPOCH):

		nerfnet_coarse.train()
		nerfnet_fine.train()
		tq = tqdm(train_dataloader)

		train_loss_tracker = []
		train_psnr_tracker = []
		# print(list(nerfnet.parameters())[0])

		for idx, (image, c2wMatrix) in enumerate(tq, start=1):

			optimizer.zero_grad()

			image, c2wMatrix = image.to(config.device), c2wMatrix.to(config.device)

			rays, t_vals = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)

			prediction   = nerfnet_coarse(rays)

			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples,\
										prediction.shape[-1]))

			rgb_coarse, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=rays, t_vals=t_vals, random_sampling=True)
			rgb_coarse = torch.permute(rgb_coarse, (0, 3, 1, 2))


			fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_vals, weights=weights)

			prediction   = nerfnet_fine(fine_rays)
			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples_fine,\
										prediction.shape[-1]))

			rgb_fine, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=fine_rays, t_vals=t_vals_fine, random_sampling=True)
			rgb_fine = torch.permute(rgb_fine, (0, 3, 1, 2))

			loss = loss_fn(image, rgb_coarse) + loss_fn(image, rgb_fine)
			loss.backward()
			optimizer.step()

			train_loss_tracker.append(loss.detach().cpu())
			train_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())

			# tq.set_description('E: {}, tr_loss: {:0.3f}'.format(epoch, loss_tracker/(idx*config.batch_size)))
			tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(train_loss_tracker)/len(train_loss_tracker), sum(train_psnr_tracker)/len(train_psnr_tracker)))

			del image, c2wMatrix, rays, t_vals, prediction, rgb_coarse, weights, fine_rays, t_vals_fine, rgb_fine, loss


		with torch.no_grad():
			val_loss_tracker = []
			val_psnr_tracker = []
			tq = tqdm(val_dataloader)

			for idx, (image, c2wMatrix) in enumerate(tq, start=1):

				image, c2wMatrix = image.to(config.device), c2wMatrix.to(config.device)

				rays, t_vals = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)

				prediction   = nerfnet_coarse(rays)

				prediction   = torch.reshape(prediction,\
											(config.batch_size,\
											config.image_height,\
											config.image_width,\
											config.num_samples,\
											prediction.shape[-1]))

				rgb_coarse, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=rays, t_vals=t_vals, random_sampling=True)
				rgb_coarse = torch.permute(rgb_coarse, (0, 3, 1, 2))


				fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_vals, weights=weights)

				prediction   = nerfnet_fine(fine_rays)

				prediction   = torch.reshape(prediction,\
											(config.batch_size,\
											config.image_height,\
											config.image_width,\
											config.num_samples_fine,\
											prediction.shape[-1]))

				rgb_fine, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=fine_rays, t_vals=t_vals_fine, random_sampling=True)
				rgb_fine = torch.permute(rgb_fine, (0, 3, 1, 2))

				loss = loss_fn(image, rgb_coarse) + loss_fn(image, rgb_fine)

				val_loss_tracker.append(loss.detach().cpu())
				val_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())

				# tq.set_description('E: {}, tr_loss: {:0.3f}'.format(epoch, loss_tracker/(idx*config.batch_size)))
				tq.set_description('E: {}, VL: {:0.3f}, VPSNR: {:0.3f}'.format(epoch, sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))


			rays, t_vals = nerf_comp.sampling_rays(camera_matrix=base_c2wMatrix, random_sampling=True)
			prediction   = nerfnet_coarse(rays)
			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples,\
										prediction.shape[-1]))
			rgb, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=rays, t_vals=t_vals, random_sampling=True)
			fine_rays, t_vals_fine  = nerf_comp.sampling_fine_rays(camera_matrix=base_c2wMatrix, t_vals=t_vals, weights=weights)
			prediction   = nerfnet_fine(fine_rays)
			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples_fine,\
										prediction.shape[-1]))
			rgb, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=fine_rays, t_vals=t_vals_fine, random_sampling=True)
			rgb = torch.permute(rgb, (0, 3, 1, 2))
			show(imgs=rgb[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
			show(imgs=depth_map[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)

		torch.save({
					'epoch': epoch,
					'model_state_dict_coarse': nerfnet_coarse.state_dict(),
					# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
					'model_state_dict_fine': nerfnet_fine.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
					# 'scheduler_state_dict': scheduler.state_dict()
			}, ckpt_path)

		with open('EXPERIMENT_{}/log.txt'.format(experiment_num), 'a') as file:
			file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}, VL: {:0.3f}, VPSNR: {:0.3f}\n'.\
				format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
				sum(train_psnr_tracker)/len(train_psnr_tracker),\
				sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
		#########################################################################################