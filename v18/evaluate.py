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
from piq import psnr, ssim, LPIPS

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
	val_dataloader = DataLoader(NerfDataLoader(camera_path=config.val_camera_path,\
										   data_path=config.val_image_path,\
										   imageHeight=config.image_height,\
										   imageWidth=config.image_width),\
										   batch_size=config.batch_size, shuffle=True,\
										   num_workers=8, pin_memory=True, drop_last=False)
	
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
	
	optimizer = torch.optim.Adam(\
									list(nerfnet_coarse.parameters()) +\
									list(nerfnet_fine.parameters()),
									lr=config.lr,
									betas=(0.9, 0.999)
								)

	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lrsch_gamma, verbose=True)
	#########################################################################################

	#########################################################################################
	dir_info  = natsorted(glob('EXPERIMENT_*'))

	experiment_num = 1

	ckpt_lst = natsorted(glob('EXPERIMENT_{}/checkpoints/nerf_*.pth'.format(experiment_num)))

	ckpt_path  = ckpt_lst[-1]
	checkpoint = torch.load(ckpt_path)
	nerfnet_coarse.load_state_dict(checkpoint['model_state_dict_coarse'])
	nerfnet_fine.load_state_dict(checkpoint['model_state_dict_fine'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	START_EPOCH = checkpoint['epoch']
	print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
	START_EPOCH += 1
	#########################################################################################

	#########################################################################################
	# nerfnet_coarse.train()
	nerfnet_fine.eval()

	tq = tqdm(val_dataloader)
	# print(list(nerfnet.parameters())[0])
	mse_loss_tracker     = []
	psnr_loss_tracker    = []
	ssim_loss_tracker    = []
	lpips_loss_tracker   = []

	for image_no, (base_image, base_c2wMatrix, base_focal, base_direction, base_near, base_far) in enumerate(tq, start=1):

		# with torch.no_grad():
		# temp_loss_tracker = [0.0]
		# temp_psnr_tracker = [0.0]
		base_image, base_c2wMatrix = torch.squeeze(base_image.to(config.device), dim=0), torch.squeeze(base_c2wMatrix.to(config.device), dim=0)
		base_focal, base_direction = torch.squeeze(base_focal.to(config.device), dim=0), torch.squeeze(base_direction.to(config.device), dim=0)
		base_near,  base_far       = torch.squeeze(base_near.to(config.device), dim=0), torch.squeeze(base_far.to(config.device), dim=0)
		base_direction             = torch.reshape(base_direction, (-1, 3))
		base_image                 = torch.reshape(base_image, (-1, 3))

		with torch.no_grad():
			rgb_final, depth_final= [], []
			for idx  in range(0, config.image_height*config.image_width, config.n_samples):
				image =  base_image[idx:idx+config.n_samples]
				ray_origin, ray_direction = nerf_comp.get_rays(base_c2wMatrix, base_direction[idx:idx+config.n_samples])
				if config.use_ndc:
					ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction, base_near, base_far, base_focal)
				view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
				view_direction_c  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples, 1]), config.dir_enc_dim)
				view_direction_f  = nerf_comp.encode_position(torch.tile(view_direction, [1, config.num_samples_fine + config.num_samples, 1]), config.dir_enc_dim)

				rays, t_vals     = nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=base_near, far=base_far, random_sampling=True)

				rgb, density   = nerfnet_coarse(rays, view_direction_c)

				rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=config.noise_value, random_sampling=True)

				# rgb_final.append(rgb_coarse)
				# depth_final.append(depth_map_coarse)

				fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse)

				rgb, density   = nerfnet_fine(fine_rays, view_direction_f)

				rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=config.noise_value, random_sampling=True)

				rgb_final.append(rgb_fine)
				depth_final.append(depth_map_fine)

				# del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, rgb_fine, depth_map_fine, weights_fine

			rgb_final  = torch.concat(rgb_final, dim=0).reshape(config.image_height, config.image_width, -1)
			depth_final = torch.concat(depth_final, dim=0).reshape(config.image_height, config.image_width)
			show(imgs=(torch.clip(rgb_final, 0, 1)*255.0).to(torch.uint8), path='EXPERIMENT_{}/eval_results/'.format(experiment_num), label='rgb', idx=image_no)
			show(imgs=depth_final, path='EXPERIMENT_{}/eval_results/'.format(experiment_num), label='depth', idx=image_no)

			base_image = torch.reshape(base_image, [config.image_height, config.image_width, -1])
			rgb_final  = torch.unsqueeze(torch.permute(rgb_final, [2, 0, 1]), dim=0)
			base_image = torch.unsqueeze(torch.permute(base_image, [2, 0, 1]), dim=0)

		mse_loss_tracker.append(torch.mean( torch.square( base_image - rgb_final ) ).detach().cpu().numpy() )
		psnr_loss_tracker.append(psnr(base_image, rgb_final).detach().cpu().numpy())
		ssim_loss_tracker.append(ssim(base_image, rgb_final, data_range=1.0).detach().cpu().numpy())
		lpips_loss_tracker.append(LPIPS()(base_image, rgb_final).detach().cpu().numpy())

		print('Image: {}, MSE: {:0.4f}, PSNR: {:0.4f}, SSIM: {:0.4f}, LPIPS: {:0.4f}'.format(\
				image_no, mse_loss_tracker[-1], psnr_loss_tracker[-1], ssim_loss_tracker[-1], lpips_loss_tracker[-1]
			))

		with open('EXPERIMENT_{}/evaluation_log.txt'.format(experiment_num), 'a') as file:
			file.write('Image: {}, MSE: {:0.4f}, PSNR: {:0.4f}, SSIM: {:0.4f}, LPIPS: {:0.4f}\n'.format(\
				image_no, mse_loss_tracker[-1], psnr_loss_tracker[-1], ssim_loss_tracker[-1], lpips_loss_tracker[-1]
			))
	

	print('MMSE: {:0.4f}, MPSNR: {:0.4f}, MSSIM: {:0.4f}, MLPIPS: {:0.4f}'.format(\
			sum(mse_loss_tracker)/len(mse_loss_tracker), sum(psnr_loss_tracker)/len(psnr_loss_tracker),\
			sum(ssim_loss_tracker)/len(ssim_loss_tracker), sum(lpips_loss_tracker)/len(lpips_loss_tracker)
		))

	with open('EXPERIMENT_{}/evaluation_log.txt'.format(experiment_num), 'a') as file:
		file.write('MMSE: {:0.4f}, MPSNR: {:0.4f}, MSSIM: {:0.4f}, MLPIPS: {:0.4f}'.format(\
				sum(mse_loss_tracker)/len(mse_loss_tracker), sum(psnr_loss_tracker)/len(psnr_loss_tracker),\
				sum(ssim_loss_tracker)/len(ssim_loss_tracker), sum(lpips_loss_tracker)/len(lpips_loss_tracker)
		))