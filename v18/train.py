# https://keras.io/examples/vision/nerf/
# https://github.com/sillsill777/NeRF-PyTorch

import os
import json
import math
import shutil
import pdb
import random
import numpy as np
import torch
import config
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from nerf_model import NerfNet
from nerf_components import NerfComponents
# from piq import psnr

import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from dataloader import NerfDataLoader
from utils import show, mse2psnr
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
	train_dataloader = DataLoader(NerfDataLoader(camera_path=config.train_camera_path,\
										   data_path=config.train_image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size, shuffle=True,\
										   num_workers=8, pin_memory=True, drop_last=False)

	val_dataloader = DataLoader(NerfDataLoader(camera_path=config.val_camera_path,\
										   data_path=config.val_image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size, shuffle=True,\
										   num_workers=8, pin_memory=True, drop_last=False)
	base_image, base_c2wMatrix, base_focal, base_direction, base_near, base_far = next(iter(val_dataloader))
	base_image, base_c2wMatrix, base_direction, base_near, base_far, base_focal = torch.squeeze(base_image.to(config.device), dim=0),\
															torch.squeeze(base_c2wMatrix.to(config.device), dim=0),\
															torch.squeeze(base_direction.to(config.device), dim=0),\
															torch.squeeze(base_near.to(config.device), dim=0),\
															torch.squeeze(base_far.to(config.device), dim=0),\
															torch.squeeze(base_focal.to(config.device), dim=0)

	print(base_image.shape, base_c2wMatrix.shape, base_direction.shape, base_near, base_far, base_focal)
	print(base_c2wMatrix)

	base_direction = torch.reshape(base_direction, (-1, 3))
	base_image     = torch.reshape(base_image, (-1, 3))
	# print(image.shape, c2wMatrix.shape)
	# show(image.to(torch.uint8))
	
	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   batch_size=config.batch_size,\
							   num_samples_coarse=config.num_samples,\
							   num_samples_fine=config.num_samples_fine,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	nerfnet_coarse = NerfNet(depth=config.net_depth, in_feat=config.in_feat, dir_feat=config.dir_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_fine   = NerfNet(depth=config.net_depth, in_feat=config.in_feat, dir_feat=config.dir_feat,\
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
									lr=config.lr,
									betas=(0.9, 0.999)
								)

	# scheduler_coarse = torch.optim.lr_scheduler.StepLR(optimizer_coarse, step_size=config.lrsch_step, gamma=config.lrsch_gamma, verbose=True)
	# scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lrsch_step, gamma=config.lrsch_gamma, verbose=True)
	# scheduler_coarse = torch.optim.lr_scheduler.ExponentialLR(optimizer_coarse, gamma=config.lrsch_gamma, verbose=True)
	# scheduler_fine   = torch.optim.lr_scheduler.ExponentialLR(optimizer_fine, gamma=config.lrsch_gamma, verbose=True)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lrsch_gamma, verbose=True)

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

	ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/nerf_*.pth'.format(experiment_num)))

	if len(ckpt_lst)>=1:
		ckpt_path  = ckpt_lst[-1]
		checkpoint = torch.load(ckpt_path)
		nerfnet_coarse.load_state_dict(checkpoint['model_state_dict_coarse'])
		nerfnet_fine.load_state_dict(checkpoint['model_state_dict_fine'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		# optimizer_coarse.load_state_dict(checkpoint['optimizer_state_dict_coarse'])
		# optimizer_fine.load_state_dict(checkpoint['optimizer_state_dict_fine'])
		# scheduler_coarse.load_state_dict(checkpoint['scheduler_state_dict_coarse'])
		# scheduler_fine.load_state_dict(checkpoint['scheduler_state_dict_fine'])
		START_EPOCH = checkpoint['epoch']
		print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
		START_EPOCH += 1
	else:
		os.makedirs('EXPERIMENT_{}/checkpoints/'.format(experiment_num))
	#########################################################################################

	#########################################################################################
	nerfnet_coarse.train()
	# nerfnet_fine.train()

	for epoch in range(START_EPOCH, END_EPOCH):

		train_loss_tracker = [0.0]
		train_psnr_tracker = [0.0]
		tq = tqdm(train_dataloader)
		# print(list(nerfnet.parameters())[0])

		for idx, (image, c2wMatrix, focal, direction, near, far) in enumerate(tq, start=1):

			# with torch.no_grad():
			# temp_loss_tracker = [0.0]
			# temp_psnr_tracker = [0.0]
			image, c2wMatrix = torch.squeeze(image.to(config.device), dim=0), torch.squeeze(c2wMatrix.to(config.device), dim=0)
			focal, direction = torch.squeeze(focal.to(config.device), dim=0), torch.squeeze(direction.to(config.device), dim=0)
			near,  far       = torch.squeeze(near.to(config.device), dim=0), torch.squeeze(far.to(config.device), dim=0)
			# if config.dataset_type == 'synthetic':
			# 	image            = torch.permute(image, (1, 2, 0))
			image            = image.reshape(-1, 3)
			direction        = direction.reshape(-1, 3)
			# random_idx       = torch.randint(low=0, high=config.image_height*config.image_width, size=(config.n_samples,))
			if epoch < config.pre_epoch:
				p  = np.ones(shape=(config.image_height, config.image_width), dtype=np.float64)
				dH = int(0.5 * config.image_height * config.pre_crop)
				dW = int(0.5 * config.image_width * config.pre_crop)
				p[config.image_height//2-dH:config.image_height//2+dH+1, config.image_width//2-dW:config.image_width//2+dW+1] = 1000.0
				p = p.reshape(-1)
				p /= p.sum()
				random_idx   = np.random.choice(a=config.image_height*config.image_width, size=[config.n_samples], replace=False, p=p)
			else:
				random_idx   = np.random.choice(a=config.image_height*config.image_width, size=[config.n_samples], replace=False)
			image            = image[random_idx]
			direction        = direction[random_idx]
			
			# calculating camera origin and ray direction
			ray_origin, ray_direction = nerf_comp.get_rays(c2wMatrix, direction)

			if config.use_ndc:
				ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction, near, far, focal)

			view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
			view_direction_c  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples, 1]), config.dir_enc_dim)
			view_direction_f  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples_fine + config.num_samples, 1]), config.dir_enc_dim)
			rays, t_vals      = nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=near, far=far, random_sampling=True)

			# optimizer_coarse.zero_grad()
			# optimizer_fine.zero_grad()

			rgb, density   = nerfnet_coarse(rays, view_direction_c)

			rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=config.noise_value, random_sampling=True)

			# with torch.no_grad():
			fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse)

			rgb, density   = nerfnet_fine(fine_rays, view_direction_f)
			
			rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=config.noise_value, random_sampling=True)

			optimizer.zero_grad()

			loss = torch.mean( torch.square(image - rgb_coarse) ) + torch.mean( torch.square(image - rgb_fine) )

			# loss = torch.mean( torch.square(image - rgb_coarse) )

			loss.backward()
			# optimizer_coarse.step()
			# optimizer_fine.step()
			optimizer.step()

			# temp_loss_tracker.append(loss.detach().cpu())
			# temp_psnr_tracker.append(mse2psnr(loss).detach().cpu())
			# train_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())
			train_loss_tracker.append(loss.detach().cpu())
			train_psnr_tracker.append(mse2psnr(torch.mean( torch.square(image - rgb_fine) ) + 1e-6).detach().cpu())

			tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(train_loss_tracker)/len(train_loss_tracker), sum(train_psnr_tracker)/len(train_psnr_tracker)))
			del rgb_coarse, depth_map_coarse, weights_coarse, fine_rays, t_vals_fine, rgb, density, rgb_fine, depth_map_fine, weights_fine, loss, view_direction_c, view_direction_f, direction, near, far, ray_origin, view_direction
			# del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, loss, view_direction_c, view_direction_f, view_direction
		# 	# break

		# # tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(train_loss_tracker)/len(train_loss_tracker), sum(train_psnr_tracker)/len(train_psnr_tracker)))
		# # break
			


		# with torch.no_grad():
		# 	val_loss_tracker = [0.0]
		# 	val_psnr_tracker = [0.0]
		# 	tq = tqdm(val_dataloader)
		# 	# print(list(nerfnet.parameters())[0])

		# 	for idx, (image, c2wMatrix, _) in enumerate(tq, start=1):

		# 		temp_loss_tracker = [0.0]

		# 		image, c2wMatrix = image.to(config.device), c2wMatrix.to(config.device)
		# 		image            = torch.permute(image, (0, 2, 3, 1))
		# 		rays, t_vals     = nerf_comp.sampling_rays(camera_matrix=c2wMatrix, random_sampling=True)

		# 		image  = image.reshape(config.batch_size, -1, 3)


		# 		rays, t_vals, image  = nerf_comp.sub_batching(rays, t_vals, image, chunk_size=config.chunk_size)

		# 		for idx, (ray_chunk, t_val_chunk, image_chunk) in enumerate(zip(rays, t_vals, image)):

		# 			optimizer.zero_grad()

		# 			prediction   = nerfnet_coarse(ray_chunk)

		# 			# prediction   = torch.reshape(prediction,\
		# 			# 							(config.batch_size,\
		# 			# 							config.image_height//config.chunk_size,\
		# 			# 							config.image_width,\
		# 			# 							config.num_samples,\
		# 			# 							prediction.shape[-1]))

		# 			rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(prediction=prediction, rays=ray_chunk, t_vals=t_val_chunk, random_sampling=True)

		# 			# rgb_coarse = torch.permute(rgb_coarse, (0, 3, 1, 2))

		# 			fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(camera_matrix=c2wMatrix, t_vals=t_val_chunk, weights=weights_coarse, idx=idx, chunk_size=config.chunk_size)

		# 			prediction   = nerfnet_fine(fine_rays)
		# 			# prediction   = torch.reshape(prediction,\
		# 			# 							(config.batch_size,\
		# 			# 							config.image_height,\
		# 			# 							config.image_width,\
		# 			# 							config.num_samples_fine,\
		# 			# 							prediction.shape[-1]))

		# 			rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(prediction=prediction, rays=fine_rays, t_vals=t_vals_fine, random_sampling=True)

		# 			# rgb_fine = torch.permute(rgb_fine, (0, 3, 1, 2))

		# 			loss = loss_fn(image_chunk, rgb_coarse) + loss_fn(image_chunk, rgb_fine)

		# 			temp_loss_tracker.append(loss.detach().cpu())
		# 			# val_psnr_tracker.append(psnr(rgb_fine, image).detach().cpu())
		# 			val_loss_tracker.append(sum(temp_loss_tracker)/len(temp_loss_tracker))

		# 			# tq.set_description('E: {}, tr_loss: {:0.3f}'.format(epoch, loss_tracker/(idx*config.batch_size)))

		# 			# 	del image, c2wMatrix, rays, t_vals, prediction, rgb_coarse, weights, fine_rays, t_vals_fine, rgb_fine, loss
		# 			# break

		# 		# tq.set_description('E: {}, TL: {:0.3f}, TPSNR: {:0.3f}'.format(epoch, sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
		# 		tq.set_description('E: {}, VL: {:0.3f}'.format(epoch, sum(val_loss_tracker)/len(val_loss_tracker)))
		# 		# break

		test_psnr = [0.0]
		if (epoch%config.vis_freq) == 0:
			with torch.no_grad():
				rgb_final, depth_final= [], []
				# image  = torch.permute(base_image, (0, 2, 3, 1))
				# image  = image.reshape(config.batch_size, -1, 3)

				for idx  in range(0, config.image_height*config.image_width, config.n_samples):
					image =  base_image[idx:idx+config.n_samples]

					ray_origin, ray_direction = nerf_comp.get_rays(base_c2wMatrix, base_direction[idx:idx+config.n_samples])
					if config.use_ndc:
						ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction, base_near, base_far, base_focal)
					view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
					view_direction_c  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples, 1]), config.dir_enc_dim)
					view_direction_f  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples_fine + config.num_samples, 1]), config.dir_enc_dim)
					# ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction)

					rays, t_vals     = nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=base_near, far=base_far, random_sampling=True)

					rgb, density   = nerfnet_coarse(rays, view_direction_c)

					rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=config.noise_value, random_sampling=True)

					# rgb_final.append(rgb_coarse)
					# depth_final.append(depth_map_coarse)

					fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse)

					rgb, density   = nerfnet_fine(fine_rays, view_direction_f)

					rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=config.noise_value, random_sampling=True)

					test_psnr.append( mse2psnr(torch.mean( torch.square(image - rgb_fine) ) + 1e-6 ).detach().cpu())

					rgb_final.append(rgb_fine)
					depth_final.append(depth_map_fine)

					del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, rgb_fine, depth_map_fine, weights_fine

				rgb_final = torch.concat(rgb_final, dim=0).reshape(config.image_height, config.image_width, -1)
				rgb_final = (torch.clip(rgb_final, 0, 1)*255.0).to(torch.uint8)
				depth_final = torch.concat(depth_final, dim=0).reshape(config.image_height, config.image_width)

			show(imgs=rgb_final, path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
			show(imgs=depth_final, path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)
			print('Epoch: {}, Test PSNR: {:0.3f}'.format(epoch, sum(test_psnr)/len(test_psnr)))
			# del rgb_final, depth_final, rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, view_direction
			del rgb_final, depth_final

		if (epoch%config.ckpt_freq)==0:
			torch.save({
						'epoch': epoch,
						'model_state_dict_coarse': nerfnet_coarse.state_dict(),
						# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
						'model_state_dict_fine': nerfnet_fine.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
						# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
						# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
						'scheduler_state_dict': scheduler.state_dict(),
						# 'scheduler_state_dict_coarse': scheduler_coarse.state_dict(),
						# 'scheduler_state_dict_fine': scheduler_fine.state_dict()
				}, 'EXPERIMENT_{}/checkpoints/nerf_{}.pth'.format(experiment_num, epoch))

		with open('EXPERIMENT_{}/log.txt'.format(experiment_num), 'a') as file:
			# file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}, VL: {:0.3f}, VPSNR: {:0.3f}\n'.\
			# 	format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
			# 	sum(train_psnr_tracker)/len(train_psnr_tracker),\
			# 	sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
			file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}, Test PSNR: {:0.3f}\n'.\
				format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
				sum(train_psnr_tracker)/len(train_psnr_tracker), sum(test_psnr)/len(test_psnr)))
		
		if (epoch%config.lrsch_step)==0:
			# scheduler_coarse.step()
			# scheduler_fine.step()
			scheduler.step()
		# scheduler.step()
		# scheduler_coarse.step()
		# scheduler_fine.step()
		# #########################################################################################
