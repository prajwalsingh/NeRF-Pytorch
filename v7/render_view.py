# https://keras.io/examples/vision/nerf/
import os
import torch
import config
import numpy as np
from glob import glob
from natsort import natsorted
from nerf_model import NerfNet
from nerf_components import NerfComponents
from tqdm import tqdm
import imageio

# Getting the translation matrix for translation t
def get_tranlation_matrix_t(t):
	matrix = [
				[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, t],
				[0, 0, 0, 1],
			 ]
	return torch.as_tensor(matrix, dtype=torch.float32)

# Getting the rotation matrix, rotates along y-axis (theta x-z plane)
def get_rotation_matrix_theta(theta):
	if isinstance(theta, float):
		theta  = torch.as_tensor([theta], dtype=torch.float32)
	matrix = [
				[torch.cos(theta), 0, -torch.sin(theta), 0],
				[0, 1, 0, 0],
				[torch.sin(theta), 0, torch.cos(theta), 0],
				[0, 0, 0, 1]
			 ]
	return torch.as_tensor(matrix, dtype=torch.float32)

# Getting the rotation matrix, rotates along x-axis (phi y-z plane)
def get_rotation_matrix_phi(phi):
	if isinstance(phi, float):
		phi  = torch.as_tensor([phi], dtype=torch.float32)
	matrix = [
				[1, 0, 0, 0],
				[0, torch.cos(phi), -torch.sin(phi), 0],
				[0, torch.sin(phi), torch.cos(phi), 0],
				[0, 0, 0, 1],
			 ]
	return torch.as_tensor(matrix, dtype=torch.float32)


# Transforming camera to world coordinates
def spherical_pose(theta, phi, t):
	c2w = get_tranlation_matrix_t(t)
	c2w = get_rotation_matrix_phi( (phi / 180.0) * np.pi) @ c2w
	c2w = get_rotation_matrix_theta(theta / 180.0 * np.pi) @ c2w
	# Why below step??
	c2w = torch.as_tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), dtype=torch.float32) @ c2w
	return c2w


if __name__ == '__main__':

	# print(get_tranlation_matrix_t(0.1))
	# print(get_rotation_matrix_theta(0.2))
	# print(get_rotation_matrix_phi(0.15))
	# jsonPath = 'dataset/nerf_synthetic/ship/transforms_val.json'
	# with open(jsonPath, 'r') as file:
	# 	jsonData = json.load(file)

	focal = torch.as_tensor([177.77], dtype=torch.float32)
	rgb_frames = []

	nerfnet_coarse = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_fine   = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_coarse = torch.nn.DataParallel(nerfnet_coarse).to(config.device)
	nerfnet_fine   = torch.nn.DataParallel(nerfnet_fine).to(config.device)

	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   focal=focal.to(config.device),\
							   batch_size=config.batch_size,\
							   near=config.near_plane,\
							   far=config.far_plane,\
							   num_samples=config.num_samples,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	#########################################################################################
	# dir_info  = natsorted(glob('EXPERIMENT_*'))

	# if len(dir_info)==0:
	# 	experiment_num = 1
	# else:
	# 	experiment_num = int(dir_info[-1].split('_')[-1]) #+ 1

	# if not os.path.isdir('EXPERIMENT_{}'.format(experiment_num)):
	# 	os.makedirs('EXPERIMENT_{}'.format(experiment_num))

	# os.system('cp *.py EXPERIMENT_{}'.format(experiment_num))
	label = 'drums'
	experiment_num = 13
	ckpt_path  = natsorted(glob('EXPERIMENT_{}/checkpoints/nerf_*.pth'.format(experiment_num)))[-1]

	if os.path.isfile(ckpt_path):		
		checkpoint = torch.load(ckpt_path)
		nerfnet_coarse.load_state_dict(checkpoint['model_state_dict_coarse'])
		# optimizer_coarse.load_state_dict(checkpoint['optimizer_state_dict_coarse'])
		nerfnet_fine.load_state_dict(checkpoint['model_state_dict_fine'])
		# optimizer_fine.load_state_dict(checkpoint['optimizer_state_dict_fine'])
		# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		# scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		START_EPOCH = checkpoint['epoch']
		print('Loading checkpoint from previous epoch: {}'.format(START_EPOCH))
		START_EPOCH += 1
	#########################################################################################

	for idx, theta in enumerate(tqdm(np.linspace(0.0, 360.0, 120, endpoint=False))):
		# Camera to world matrix
		with torch.no_grad():
			c2w = torch.unsqueeze(spherical_pose(theta, -30.0, 4.0), dim=0)

			rays, t_vals = nerf_comp.sampling_rays(camera_matrix=c2w, random_sampling=True)
			prediction   = nerfnet_coarse(rays)
			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples,\
										prediction.shape[-1]))
			rgb, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=rays, t_vals=t_vals, random_sampling=True)
			# torch.cuda.empty_cache()
			fine_rays, t_vals_fine  = nerf_comp.sampling_fine_rays(camera_matrix=c2w, t_vals=t_vals, weights=weights)
			prediction   = nerfnet_fine(fine_rays)
			prediction   = torch.reshape(prediction,\
										(config.batch_size,\
										config.image_height,\
										config.image_width,\
										config.num_samples_fine,\
										prediction.shape[-1]))
			rgb, depth_map, weights = nerf_comp.render_rgb_depth(prediction=prediction, rays=fine_rays, t_vals=t_vals_fine, random_sampling=True)
			# rgb = torch.permute(rgb, (0, 3, 1, 2))
			# show(imgs=rgb[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
			# show(imgs=depth_map[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)
			rgb_frames = rgb_frames + [ np.uint8(np.clip(rgb[0].detach().cpu().numpy()*255.0, 0, 255)) ]

	rgb_video = "EXPERIMENT_{}/{}.mp4".format(experiment_num, label)
	imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)