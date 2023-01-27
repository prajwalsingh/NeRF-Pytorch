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
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams["savefig.bbox"] = 'tight'
# plt.rcParams["figure.figsize"] = (18,18)
style.use('seaborn')

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
	# focal_len  = 600.0
	# base_focal = torch.as_tensor([focal_len], dtype=torch.float32).to(config.device)
	# base_near  = torch.as_tensor([3.3], dtype=torch.float32).to(config.device)
	# base_far   = torch.as_tensor([28.0], dtype=torch.float32).to(config.device)
	datatype   = 'synthetic'
	# datatype   = 'real'
	focal_len  = 600.0
	base_focal = torch.as_tensor([focal_len], dtype=torch.float32).to(config.device)
	base_near  = torch.as_tensor([2.0], dtype=torch.float32).to(config.device)
	base_far   = torch.as_tensor([6.0], dtype=torch.float32).to(config.device)

	x, y = np.meshgrid(
							np.arange(0, config.image_width, dtype=np.float32),
							np.arange(0, config.image_height, dtype=np.float32),
							indexing = 'xy'
						)

	# Pixel to camera coordinates
	camera_x = (x - (config.image_width  * 0.5))/focal_len
	camera_y = (y - (config.image_height * 0.5))/focal_len

	# creating a direction vector and normalizing to unit vector
	# direction of pixels w.r.t local camera origin (0,0,0)

	base_direction = torch.FloatTensor(np.stack([camera_x, -camera_y, -np.ones_like(camera_x)], axis=-1)).to(config.device)

	base_direction = torch.reshape(base_direction, [-1, 3])

	rgb_frames = []
	depth_frames = []

	nerfnet_coarse = NerfNet(depth=config.net_depth, in_feat=config.in_feat, dir_feat=config.dir_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_fine   = NerfNet(depth=config.net_depth, in_feat=config.in_feat, dir_feat=config.dir_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(config.device)

	nerfnet_coarse = torch.nn.DataParallel(nerfnet_coarse).to(config.device)
	nerfnet_fine   = torch.nn.DataParallel(nerfnet_fine).to(config.device)

	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   batch_size=config.batch_size,\
							   num_samples_coarse=config.num_samples,\
							   num_samples_fine=config.num_samples_fine,\
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
	label = 'lego'
	experiment_num = 3
	ckpt_path  = natsorted(glob('EXPERIMENT_{}/checkpoints/nerf_*.pth'.format(experiment_num)))[-1]

	if not os.path.isdir('EXPERIMENT_{}/results/'.format(experiment_num)):
		os.makedirs('EXPERIMENT_{}/results/rgb'.format(experiment_num))
		os.makedirs('EXPERIMENT_{}/results/depth'.format(experiment_num))

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

	for i, theta in enumerate(tqdm(np.linspace(0.0, 360.0, 120, endpoint=False))):
		with torch.no_grad():
			# Camera to world matrix
			base_c2wMatrix = spherical_pose(theta, -30.0, 4.0).to(config.device)

			rgb_final, depth_final = [], []				
			# image  = torch.permute(base_image, (0, 2, 3, 1))
			# image  = image.reshape(config.batch_size, -1, 3)

			for idx  in range(0, config.image_height*config.image_width, config.n_samples):
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
				del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f

			rgb_final = torch.concat(rgb_final, dim=0).reshape(config.image_height, config.image_width, -1)
			# rgb_final = torch.permute(rgb_final, (0, 3, 1, 2))
			depth_final = torch.concat(depth_final, dim=0).reshape(config.image_height, config.image_width)

			# rgb = torch.permute(rgb, (0, 3, 1, 2))
			# show(imgs=rgb[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='img', idx=epoch)
			# show(imgs=depth_map[:1], path='EXPERIMENT_{}/train'.format(experiment_num), label='depth', idx=epoch)
			IMG = np.uint8(np.clip(rgb_final.detach().cpu().numpy()*255.0, 0, 255))
			# DEPTH = np.uint8(np.clip(depth_final.detach().cpu().numpy()*255.0, 0, 255))
			DEPTH = depth_final.detach().cpu().numpy()
			DEPTH = (DEPTH - np.min(DEPTH)) / (np.max(DEPTH) - np.min(DEPTH))
			DEPTH = np.uint8(np.clip(DEPTH*255.0, 0, 255))
			rgb_frames = rgb_frames + [ IMG ]
			depth_frames = depth_frames + [ DEPTH ]
			plt.figure(figsize=(9, 9), dpi=96)
			plt.imshow(IMG)
			plt.axis('off')
			plt.grid(False)
			plt.savefig('EXPERIMENT_{}/results/rgb/{}.png'.format(experiment_num, i))
			plt.close()
			plt.figure(figsize=(9, 9), dpi=96)
			plt.imshow(DEPTH, cmap='gray')
			plt.axis('off')
			plt.grid(False)
			plt.savefig('EXPERIMENT_{}/results/depth/{}.png'.format(experiment_num, i))
			plt.close()

	rgb_video = "EXPERIMENT_{}/results/{}_rgb.mp4".format(experiment_num, label)
	depth_video = "EXPERIMENT_{}/results/{}_depth.mp4".format(experiment_num, label)
	imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)
	imageio.mimwrite(depth_video, depth_frames, fps=30, quality=7, macro_block_size=None)