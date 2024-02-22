# https://keras.io/examples/vision/nerf/
# https://github.com/sillsill777/NeRF-PyTorch

import os
import pdb
import json
import math
import torch
import shutil
import random
import argparse
import numpy as np
import torch.nn as nn
from nerf_model import NerfNet, NerfNetLight
from torchvision import transforms, io
from nerf_components import NerfComponents
from torch.utils.data import Dataset, DataLoader
# from piq import psnr

import matplotlib.pyplot as plt
from matplotlib import style
from tqdm import tqdm
from dataloader import NerfDataLoader, NerfDataLoaderLight
from utils import show, mse2psnr
from glob import glob
from natsort import natsorted
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint


torch.autograd.set_detect_anomaly(True)
plt.rcParams["savefig.bbox"] = 'tight'
# style.use('seaborn')
# torch.manual_seed(45)
np.random.seed(45)
random.seed(45)
seed_everything(42, workers=True)

def parse_args():
	
	parser = argparse.ArgumentParser('arguments NeRF base training')
	
	parser.add_argument('--basedir', type=str, default=None, help='dataset folder location')
	parser.add_argument('--dataset_name', type=str, default=None, help='name of the dataset')
	parser.add_argument('--dataset_type', type=str, default='synthetic', help='dataset type: [synthetic, real, llff]')
	parser.add_argument('--batch_size', type=int, default=1, help='batch size to use for training/inference')
	parser.add_argument('--image_height', type=int, default=800, help='dataset image height')
	parser.add_argument('--image_width', type=int, default=800, help='dataset image width')
	parser.add_argument('--channels', type=int, default=3, help='number of channels image have')
	parser.add_argument('--scale', type=int, default=2, help='scale down factor for dataset images')
	parser.add_argument('--near_plane', type=float, default=2.0, help='near plane of the camera')
	parser.add_argument('--far_plane', type=float, default=6.0, help='far plane of the camera')
	parser.add_argument('--spherify', type=bool, default=True, help='sphere camera angle for real | llff datasets')
	parser.add_argument('--pre_crop', type=float, default=0.5, help='data augmentation crop percentage')
	parser.add_argument('--noise_value', type=float, default=0.0, help='noise value for density estimation')
	
	parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
	parser.add_argument('--lrsch_step', type=int, default=1800, help='learning rate scheduler steps')
	parser.add_argument('--lrsch_gamma', type=float, default=0.1, help='learning rate scheduler gamma')

	parser.add_argument('--experiment_num', type=int, default=1, help='current experiment number')
	parser.add_argument('--epochs', type=int, default=2001, help='total epochs')
	parser.add_argument('--pre_epoch', type=int, default=50, help='Warmup epochs')
	parser.add_argument('--vis_freq', type=int, default=10, help='run validation step')
	parser.add_argument('--ckpt_freq', type=int, default=10, help='save checkpoints after every n epochs')
	parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus available')
	parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory for fast dataloading')
	
	parser.add_argument('--pos_enc_dim', type=int, default=10, help='position encoding k')
	parser.add_argument('--dir_enc_dim', type=int, default=4, help='directional encoding k')

	parser.add_argument('--num_samples', type=int, default=64, help='number of points to sample in coarse')
	parser.add_argument('--num_samples_fine', type=int, default=128, help='number of points to sample in fine')
	parser.add_argument('--net_dim', type=int, default=256, help='network layer size')
	parser.add_argument('--skip_layer', type=int, default=4, help='after how many layers to concat direction encoding')
	parser.add_argument('--net_depth', type=int, default=8, help='number of fully connected layers')
	parser.add_argument('--n_samples', type=int, default=4096, help='number of rays to sample from each image')


	args = parser.parse_args()

	return args



if __name__ == '__main__':

	args = parse_args()

	args.in_feat      = 2*(args.channels*args.pos_enc_dim) + args.channels
	args.dir_feat     = 2*(args.channels*args.dir_enc_dim) + args.channels

	nerf_comp = NerfComponents(height=args.image_height,\
							   width=args.image_width,\
							   batch_size=args.batch_size,\
							   num_samples_coarse=args.num_samples,\
							   num_samples_fine=args.num_samples_fine,\
							   pos_enc_dim=args.pos_enc_dim,\
							   dir_enc_dim=args.dir_enc_dim)

	nerfnet_coarse = NerfNet(depth=args.net_depth, in_feat=args.in_feat, dir_feat=args.dir_feat,\
					  net_dim=args.net_dim, skip_layer=args.skip_layer)

	nerfnet_fine   = NerfNet(depth=args.net_depth, in_feat=args.in_feat, dir_feat=args.dir_feat,\
					  net_dim=args.net_dim, skip_layer=args.skip_layer)

	# if os.getenv("LOCAL_RANK", '0') == '0':

	# 	global save_path 

	# 	dir_info  = natsorted(glob('run/*_experiment*'))

	# 	if len(dir_info)==0:
	# 		experiment_num = 1
	# 	else:
	# 		experiment_num = int(os.path.split(dir_info[-1])[-1].split('_')[0]) + 1
		
	save_path = 'run/{}_experiment_{}'.format(args.experiment_num, args.dataset_name)

	if not os.path.isdir(save_path):
		os.makedirs(save_path)
		os.system('cp *.py run/{}'.format(save_path))


	# saves top-K checkpoints based on "val_loss" metric
	checkpoint_callback = ModelCheckpoint(
		save_top_k=10,
		monitor="val_psnr",
		mode="max",
		dirpath="{}/checkpoints/".format(save_path),
		filename="base-nerf-{epoch:02d}-{val_psnr:.2f}",
	)

	tensorboard = pl_loggers.TensorBoardLogger(save_dir=save_path)
	trainer = L.Trainer(accelerator='gpu',\
					 	devices=args.gpus,\
					 	strategy="ddp",\
						max_epochs=args.epochs,\
						precision="16-mixed",\
						check_val_every_n_epoch=args.ckpt_freq,\
						callbacks=[checkpoint_callback],\
						logger=tensorboard,)

	light_network = NerfNetLight(nerfcomp=nerf_comp,\
							     nerfnet_coarse=nerfnet_coarse,\
								 nerfnet_fine=nerfnet_fine,\
								 args=args)
	
	data_loader = NerfDataLoaderLight(args)


	trainer.fit(model=light_network,\
			 	datamodule=data_loader)

	# if (epoch%args.vis_freq) == 0:
	# 	with torch.no_grad():
	# 		rgb_final, depth_final = [], []				
	# 		# image  = torch.permute(base_image, (0, 2, 3, 1))
	# 		# image  = image.reshape(args.batch_size, -1, 3)

	# 		for idx  in range(0, args.image_height*args.image_width, args.n_samples):
	# 			ray_origin, ray_direction = nerf_comp.get_rays(base_c2wMatrix, base_direction[idx:idx+args.n_samples])
	# 			if args.use_ndc:
	# 				ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction, base_near, base_far, base_focal)
	# 			view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
	# 			view_direction_c  = nerf_comp.encode_position(torch.tile(view_direction, [1, args.num_samples, 1]), args.dir_enc_dim)
	# 			view_direction_f  = nerf_comp.encode_position(torch.tile(view_direction, [1, args.num_samples_fine + args.num_samples, 1]), args.dir_enc_dim)
	# 			# ray_origin, ray_direction = nerf_comp.ndc_rays(ray_origin, ray_direction)

	# 			rays, t_vals     = nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=base_near, far=base_far, random_sampling=True)

	# 			rgb, density   = nerfnet_coarse(rays, view_direction_c)

	# 			rgb_coarse, depth_map_coarse, weights_coarse = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=args.noise_value, random_sampling=True)

	# 			# rgb_final.append(rgb_coarse)
	# 			# depth_final.append(depth_map_coarse)

	# 			fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse)

	# 			rgb, density   = nerfnet_fine(fine_rays, view_direction_f)

	# 			rgb_fine, depth_map_fine, weights_fine = nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=args.noise_value, random_sampling=True)

	# 			rgb_final.append(rgb_fine)
	# 			depth_final.append(depth_map_fine)

	# 			del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, rgb_fine, depth_map_fine, weights_fine

	# 		rgb_final = torch.concat(rgb_final, dim=0).reshape(args.image_height, args.image_width, -1)
	# 		rgb_final = (torch.clip(rgb_final, 0, 1)*255.0).to(torch.uint8)
	# 		depth_final = torch.concat(depth_final, dim=0).reshape(args.image_height, args.image_width)

	# 	show(imgs=rgb_final, path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
	# 	show(imgs=depth_final, path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)
	# 	# del rgb_final, depth_final, rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, view_direction
	# 	del rgb_final, depth_final

	# if (epoch%args.ckpt_freq)==0:
	# 	torch.save({
	# 				'epoch': epoch,
	# 				'model_state_dict_coarse': nerfnet_coarse.state_dict(),
	# 				# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
	# 				'model_state_dict_fine': nerfnet_fine.state_dict(),
	# 				'optimizer_state_dict': optimizer.state_dict(),
	# 				# 'optimizer_state_dict_coarse': optimizer_coarse.state_dict(),
	# 				# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
	# 				# 'optimizer_state_dict_fine': optimizer_fine.state_dict(),
	# 				'scheduler_state_dict': scheduler.state_dict(),
	# 				# 'scheduler_state_dict_coarse': scheduler_coarse.state_dict(),
	# 				# 'scheduler_state_dict_fine': scheduler_fine.state_dict()
	# 		}, 'EXPERIMENT_{}/checkpoints/nerf_{}.pth'.format(experiment_num, 'all'))

	# with open('EXPERIMENT_{}/log.txt'.format(experiment_num), 'a') as file:
	# 	# file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}, VL: {:0.3f}, VPSNR: {:0.3f}\n'.\
	# 	# 	format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
	# 	# 	sum(train_psnr_tracker)/len(train_psnr_tracker),\
	# 	# 	sum(val_loss_tracker)/len(val_loss_tracker), sum(val_psnr_tracker)/len(val_psnr_tracker)))
	# 	file.write('Epoch: {}, TL: {:0.3f}, TPSNR: {:0.3f}\n'.\
	# 		format(epoch, sum(train_loss_tracker)/len(train_loss_tracker),\
	# 		sum(train_psnr_tracker)/len(train_psnr_tracker)))
	
	# if (epoch%args.lrsch_step)==0:
	# 	# scheduler_coarse.step()
	# 	# scheduler_fine.step()
	# 	scheduler.step()
	# # scheduler.step()
	# # scheduler_coarse.step()
	# # scheduler_fine.step()
	# # #########################################################################################