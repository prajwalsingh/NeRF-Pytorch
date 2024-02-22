# References:
# https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html

import math
import random
import torch
import numpy as np
import config
import torch.nn as nn
import lightning as L
from utils import mse2psnr
from piq import psnr, ssim

torch.manual_seed(45)
np.random.seed(45)
random.seed(45)

class NerfNet(nn.Module):

	def __init__(self, depth, in_feat, dir_feat, net_dim=128, skip_layer=4):
		super(NerfNet, self).__init__()
		self.depth = depth
		self.skip_layer = skip_layer
		units = [in_feat] + [net_dim]*(self.depth+1)
		self.layers = nn.ModuleList([])
		self.bnorm_layers = nn.ModuleList([])

		# self.act    = nn.ReLU()
		# self.act     = nn.SiLU()
		# self.act    = nn.GELU()
		# self.act_out = nn.Sigmoid()

		for i in range(self.depth):
			if (i%(self.skip_layer+1)==0) and (i>0):
				self.layers.append(nn.Sequential(
								   nn.Linear(in_features=units[i]+in_feat, out_features=units[i+1]),
								   # nn.ReLU(),
								   nn.ELU(),
								#    nn.SiLU(),
								# nn.GELU(),
								#    nn.InstanceNorm1d(num_features=units[i+1]),
								   ))
				# self.layers.append(nn.Linear(in_features=units[i]+in_feat, out_features=units[i+1]))
				# self.bnorm_layers.append(nn.InstanceNorm1d(num_features=units[i+1]))
			else:
				self.layers.append(nn.Sequential(
								   nn.Linear(in_features=units[i], out_features=units[i+1]),
								   # nn.ReLU(),
								   nn.ELU(),
								#    nn.SiLU(),
								# nn.GELU(),
								#    nn.InstanceNorm1d(num_features=units[i+1]),
								   ))
				# self.layers.append(nn.Linear(in_features=units[i], out_features=units[i+1]))
				# self.bnorm_layers.append(nn.InstanceNorm1d(num_features=units[i+1]))

		self.density = nn.Sequential(
						nn.Linear(in_features=net_dim, out_features=1),
					)
		# self.density = nn.Linear(in_features=net_dim, out_features=1)
		self.feature = nn.Sequential(
						nn.Linear(in_features=net_dim, out_features=net_dim),
					)
		# self.feature = nn.Linear(in_features=net_dim, out_features=net_dim)
		self.layer_9 = nn.Sequential(
						nn.Linear(in_features=net_dim+dir_feat, out_features=net_dim//2),
						# nn.ReLU(),
						nn.ELU(),
						# nn.SiLU(),
						# nn.GELU(),
						# nn.InstanceNorm1d(num_features=units[i+1]),
					)
		# self.layer_9 = nn.Linear(in_features=net_dim+dir_feat, out_features=net_dim//2)
		self.color  = nn.Sequential(
						nn.Linear(in_features=net_dim//2, out_features=3),
					)
		# self.color   = nn.Linear(in_features=net_dim//2, out_features=3)


	def forward(self, inp, vdir):
		
		inp_n_rays, inp_n_samples, inp_c = inp.shape
		vdir_n_rays, vdir_n_samples, vdir_c = vdir.shape
		inp  = torch.reshape(inp, [-1, inp_c])
		vdir = torch.reshape(vdir, [-1, vdir_c])
		x    = inp

		for i in range(self.depth):

			# x = self.act(self.bnorm_layers[i]( self.layers[i]( x )) )
			# x = self.act( self.layers[i]( x ) )
			x = self.layers[i]( x )

			if (i%self.skip_layer==0) and (i>0):
				x = torch.concat([inp, x], dim=-1)

		# sigma = self.act_out( self.density( x ) )
		sigma = self.density( x )

		# x = self.act( self.feature( x ) )
		x = self.feature( x )

		x = torch.concat([x, vdir], dim=-1)

		# x = self.act( self.layer_9( x ) )
		x = self.layer_9( x )

		# rgb = self.act_out( self.color( x ) )
		rgb = self.color( x )

		# print('omin: {}, omax: {}'.format(out.min(), out.max()))
		sigma = torch.reshape(sigma, [-1, inp_n_samples, 1])
		rgb   = torch.reshape(rgb, [-1, inp_n_samples, 3])

		return rgb, sigma
		

class NerfNetLight(L.LightningModule):
	def __init__(self, nerfcomp, nerfnet_coarse, nerfnet_fine, args=None):
		super().__init__()
		self.save_hyperparameters(ignore=['nerfnet_coarse', 'nerfnet_fine'])

		self.nerf_comp = nerfcomp
		self.nerfnet_coarse = nerfnet_coarse
		self.nerfnet_fine  = nerfnet_fine
		self.args = args

		# Important: This property activates manual optimization.
		self.automatic_optimization = False

	def training_step(self, batch, batch_idx):
		opt = self.optimizers()
		scheduler = self.lr_schedulers()

		image, c2wMatrix, focal, direction, near, far = batch
		image, c2wMatrix, focal, direction, near, far = torch.squeeze(image, dim=0),\
														torch.squeeze(c2wMatrix, dim=0),\
														torch.squeeze(focal, dim=0),\
														torch.squeeze(direction, dim=0),\
														torch.squeeze(near, dim=0),\
														torch.squeeze(far, dim=0)

		image     = image.reshape(-1, 3)
		direction = direction.reshape(-1, 3)

		image_width  = self.args.image_width // self.args.scale
		image_height = self.args.image_height // self.args.scale
		# random_idx       = torch.randint(low=0, high=args.image_height*args.image_width, size=(args.n_samples,))
		if self.trainer.current_epoch < self.args.pre_epoch:
			p  = np.ones(shape=(image_height, image_width), dtype=np.float64)
			dH = int(0.5 * image_height * self.args.pre_crop)
			dW = int(0.5 * image_width * self.args.pre_crop)
			p[image_height//2-dH:image_height//2+dH+1, image_width//2-dW:image_width//2+dW+1] = 1000.0
			p = p.reshape(-1)
			p /= p.sum()
			random_idx   = np.random.choice(a=image_height*image_width, size=[self.args.n_samples], replace=False, p=p)
		else:
			random_idx   = np.random.choice(a=image_height*image_width, size=[self.args.n_samples], replace=False)
		image            = image[random_idx]
		direction        = direction[random_idx]
		
		# calculating camera origin and ray direction
		ray_origin, ray_direction = self.nerf_comp.get_rays(c2wMatrix, direction)

		view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
		view_direction_c  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples, 1]), self.args.dir_enc_dim)
		view_direction_f  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples_fine + self.args.num_samples, 1]), self.args.dir_enc_dim)
		rays, t_vals      = self.nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=near, far=far, random_sampling=True, device=self.device)

		# optimizer_coarse.zero_grad()
		# optimizer_fine.zero_grad()

		rgb, density   = self.nerfnet_coarse(rays, view_direction_c)

		rgb_coarse, depth_map_coarse, weights_coarse = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=self.args.noise_value, random_sampling=True, device=self.device)

		# with torch.no_grad():
		fine_rays, t_vals_fine = self.nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse, device=self.device)

		rgb, density   = self.nerfnet_fine(fine_rays, view_direction_f)
		
		rgb_fine, depth_map_fine, weights_fine = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=self.args.noise_value, random_sampling=True, device=self.device)

		opt.zero_grad()
		loss = torch.mean( torch.square(image - rgb_coarse) ) + torch.mean( torch.square(image - rgb_fine) )
		self.manual_backward(loss)
		opt.step()
		self.log('train_loss', loss, sync_dist=True)
		self.log('psnr_mse', mse2psnr(loss), sync_dist=True)

		if ((self.trainer.current_epoch + 1) % self.args.lrsch_step) == 0:
			scheduler.step()
	
	def validation_step(self, batch, batch_idx):

		image, c2wMatrix, focal, direction, near, far = batch
		image, c2wMatrix, focal, direction, near, far = torch.squeeze(image, dim=0),\
														torch.squeeze(c2wMatrix, dim=0),\
														torch.squeeze(focal, dim=0),\
														torch.squeeze(direction, dim=0),\
														torch.squeeze(near, dim=0),\
														torch.squeeze(far, dim=0)

		image_width  = self.args.image_width // self.args.scale
		image_height = self.args.image_height // self.args.scale		
		
		dH = int(0.5 * image_height * self.args.pre_crop)
		dW = int(0.5 * image_width * self.args.pre_crop)
		image      = image[image_height//2-dH:image_height//2+dH+1, image_width//2-dW:image_width//2+dW+1]
		direction  = direction[image_height//2-dH:image_height//2+dH+1, image_width//2-dW:image_width//2+dW+1]

		image     = image.reshape(-1, 3)[:self.args.n_samples]
		direction = direction.reshape(-1, 3)[:self.args.n_samples]

		# calculating camera origin and ray direction
		ray_origin, ray_direction = self.nerf_comp.get_rays(c2wMatrix, direction)

		view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
		view_direction_c  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples, 1]), self.args.dir_enc_dim)
		view_direction_f  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples_fine + self.args.num_samples, 1]), self.args.dir_enc_dim)
		rays, t_vals      = self.nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=near, far=far, random_sampling=True, device=self.device)

		rgb, density   = self.nerfnet_coarse(rays, view_direction_c)

		rgb_coarse, depth_map_coarse, weights_coarse = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=self.args.noise_value, random_sampling=True, device=self.device)

		# with torch.no_grad():
		fine_rays, t_vals_fine = self.nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse, device=self.device)

		rgb, density   = self.nerfnet_fine(fine_rays, view_direction_f)
		
		rgb_fine, depth_map_fine, weights_fine = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=self.args.noise_value, random_sampling=True, device=self.device)

		height, width = int(math.sqrt(self.args.n_samples)), int(math.sqrt(self.args.n_samples))

		rgb_fine = rgb_fine.reshape(height, width, -1)
		image    = image.reshape(height, width, -1)
		rgb_fine = torch.permute(torch.clip(rgb_fine, 0, 1), (2, 0, 1))
		image     = torch.permute(torch.clip(image, 0, 1), (2, 0, 1))

		loss = torch.mean( torch.square(image - rgb_fine) )
		psnr_metric = psnr(torch.unsqueeze(image, dim=0), torch.unsqueeze(rgb_fine, dim=0), data_range=1.)
		ssim_metric = ssim(torch.unsqueeze(image, dim=0), torch.unsqueeze(rgb_fine, dim=0), data_range=1.)

		self.log('val_loss', loss, sync_dist=True)
		self.log('val_psnr', psnr_metric, sync_dist=True)
		self.log('val_ssim', ssim_metric, sync_dist=True)		
	
	def on_validation_epoch_end(self):
		if self.trainer.global_rank == 0:
			val_dataloader = self.trainer.val_dataloaders  # one or multiple
			
			image, c2wMatrix, focal, direction, near, far = next(iter(val_dataloader))
			image, c2wMatrix, focal, direction, near, far = torch.squeeze(image, dim=0).to(self.device),\
															torch.squeeze(c2wMatrix, dim=0).to(self.device),\
															torch.squeeze(focal, dim=0).to(self.device),\
															torch.squeeze(direction, dim=0).to(self.device),\
															torch.squeeze(near, dim=0).to(self.device),\
															torch.squeeze(far, dim=0).to(self.device)
			
			rgb_final, depth_final = [], []
			image_width  = self.args.image_width // self.args.scale
			image_height = self.args.image_height // self.args.scale

			# image     = image.reshape(-1, 3)
			direction = direction.reshape(-1, 3)

			for idx  in range(0, image_height*image_width, self.args.n_samples):
				ray_origin, ray_direction = self.nerf_comp.get_rays(c2wMatrix, direction[idx:idx+self.args.n_samples])
				view_direction    = torch.unsqueeze(ray_direction / torch.linalg.norm(ray_direction, ord=2, dim=-1, keepdim=True), dim=1)
				view_direction_c  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples, 1]), self.args.dir_enc_dim)
				view_direction_f  = self.nerf_comp.encode_position(torch.tile(view_direction, [1, self.args.num_samples_fine + self.args.num_samples, 1]), self.args.dir_enc_dim)
				rays, t_vals      = self.nerf_comp.sampling_rays(ray_origin=ray_origin, ray_direction=ray_direction, near=near, far=far, random_sampling=True, device=self.device)
				rgb, density      = self.nerfnet_coarse(rays, view_direction_c)
				rgb_coarse, depth_map_coarse, weights_coarse = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals, noise_value=self.args.noise_value, random_sampling=True, device=self.device)
				fine_rays, t_vals_fine = self.nerf_comp.sampling_fine_rays(ray_origin=ray_origin, ray_direction=ray_direction, t_vals=t_vals, weights=weights_coarse, device=self.device)
				rgb, density = self.nerfnet_fine(fine_rays, view_direction_f)
				rgb_fine, depth_map_fine, weights_fine = self.nerf_comp.render_rgb_depth(rgb=rgb, density=density, rays_d=ray_direction, t_vals=t_vals_fine, noise_value=self.args.noise_value, random_sampling=True, device=self.device)
				rgb_final.append(rgb_fine)
			
				del rgb_coarse, depth_map_coarse, weights_coarse, rgb, density, view_direction_c, view_direction_f, rgb_fine, depth_map_fine, weights_fine

			rgb_final = torch.concat(rgb_final, dim=0).reshape(image_height, image_width, -1)
			rgb_final = torch.permute((torch.clip(rgb_final, 0, 1)*255.0).to(torch.uint8), (2, 0, 1))

			self.logger.experiment.add_image('val_image', rgb_final)
			

	def configure_optimizers(self):
		opt = torch.optim.Adam(\
									list(self.nerfnet_coarse.parameters()) +\
									list(self.nerfnet_fine.parameters()),
									lr=self.args.lr,
									betas=(0.9, 0.999)
								)
		scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,\
													 	   gamma=self.args.lrsch_gamma,\
														   verbose=True)
		return {"optimizer": opt, "lr_scheduler": scheduler}