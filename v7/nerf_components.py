import random
import torch
import numpy as np
import config
import pdb

torch.manual_seed(45)
np.random.seed(45)
random.seed(45)

class NerfComponents:
	def __init__(self, height, width, focal, batch_size, near, far, num_samples, pos_enc_dim, dir_enc_dim, device='cuda'):
		self.height = height
		self.width  = width
		self.focal  = focal
		self.device = device
		self.batch_size = batch_size
		self.near   = near
		self.far    = far
		self.num_samples = num_samples
		self.pos_enc_dim = pos_enc_dim
		self.dir_enc_dim = dir_enc_dim

		# Pixel of image
		x, y = torch.meshgrid(
								torch.arange(0, width, dtype=torch.float32),
								torch.arange(0, height, dtype=torch.float32),
								indexing = 'xy'
							 )
		x, y = x.to(device), y.to(device)

		# Pixel to camera coordinates
		camera_x = (x - (width  * 0.5))/focal
		camera_y = (y - (height * 0.5))/focal

		# creating a direction vector and normalizing to unit vector
		# direction of pixels w.r.t local camera origin (0,0,0)
		self.direction = torch.stack([camera_x, -camera_y, -torch.ones_like(camera_x).to(device)], axis=-1)
		self.direction = self.direction / torch.unsqueeze(torch.norm(self.direction, dim=-1), dim=-1)
		self.direction = torch.tile(torch.unsqueeze(self.direction, dim=0), (batch_size, 1, 1, 1))


	def encode_position(self, x, enc_dim):
		
		# positions = [x]
		positions = []

		for i in range(enc_dim):
			# positions.append(torch.sin( (2.0**i) * torch.pi * x ))
			# positions.append(torch.cos( (2.0**i) * torch.pi * x ))
			positions.append(torch.sin( (2.0**i) *  x ))
			positions.append(torch.cos( (2.0**i) *  x ))

		return torch.concat(positions, dim=-1).to(self.device)
	
	def ndc_rays(self, rays_o, rays_d):
		"""Normalized device coordinate rays.

		Space such that the canvas is a cube with sides [-1, 1] in each axis.

		Args:
		H: int. Height in pixels.
		W: int. Width in pixels.
		focal: float. Focal length of pinhole camera.
		near: float or array of shape[batch_size]. Near depth bound for the scene.
		rays_o: array of shape [batch_size, 3]. Camera origin.
		rays_d: array of shape [batch_size, 3]. Ray direction.

		Returns:
		rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
		rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
		"""
		# Shift ray origins to near plane
		t = -(self.near + rays_o[..., 2]) / rays_d[..., 2]
		rays_o = rays_o + t[..., None] * rays_d

		# Projection
		o0 = -1./(self.width/(2.*self.focal)) * rays_o[..., 0] / rays_o[..., 2]
		o1 = -1./(self.height/(2.*self.focal)) * rays_o[..., 1] / rays_o[..., 2]
		o2 = 1. + 2. * self.near / rays_o[..., 2]

		d0 = -1./(self.width/(2.*self.focal)) * \
			(rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
		d1 = -1./(self.height/(2.*self.focal)) * \
			(rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
		d2 = -2. * self.near / rays_o[..., 2]

		return rays_o, rays_d

	def get_rays(self, camera_matrix):

		# Rotation matrix, camera to world coordinates C_ext^{-1}
		rotation_matrix = camera_matrix[:, :3, :3].to(self.device)
		# Translation matrix from camera to world coordinates t_{ext}^{-1}
		translation     = camera_matrix[:, :3, -1].to(self.device)

		# Applying rotation matrix in left side, following is method to do so
		# without doing transpose
		# camera to world coordinates
		direction_c   = torch.unsqueeze(self.direction, dim=-2)
		rotation_matrix_c  = torch.unsqueeze(torch.unsqueeze(rotation_matrix, dim=1), dim=1)
		ray_direction      = torch.sum(direction_c * rotation_matrix_c, dim=-1)

		# Ray origin
		ray_origin         = torch.tile(torch.unsqueeze(torch.unsqueeze(translation, dim=-2), dim=-2), (1, self.height, self.width, 1))

		return (ray_origin, ray_direction)
	
	
	def sampling_rays(self, ray_origin, ray_direction, random_sampling=True):

		# Compute 3D query points
		# r(t) = o + td -> we are buildin t here [x,y,z] of raidance
		# this is discrete sampling
		t_vals = torch.linspace(0.0, 1.0, self.num_samples).to(self.device)
		z_vals = self.near * (1.0 - t_vals) + self.far * t_vals

		# continuos sampling
		if random_sampling:
			# Injecting uniform noise to make sampling continuos
			# B x H x W x num_samples
			# shape  = list(ray_origin.shape[:-1]) + [self.num_samples]
			# noise  = torch.rand(size=shape).to(self.device) * ((self.far - self.near) / self.num_samples)
			# t_vals = t_vals + noise
			mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
			upper = torch.concat([mids, z_vals[..., -1:]], dim=-1)
			lower = torch.concat([z_vals[..., :1], mids], dim=-1)
			# stratified samples in those intervals
			t_rand = torch.rand(z_vals.shape).to(config.device)
			z_vals = lower + (upper - lower) * t_rand

		# r(t) = o + td -> building "r" here
		# B x H x W x 1 x 3 + B x H x W x 1 x 3 * B x H x W x 32 x 1
		rays = torch.unsqueeze(ray_origin, dim=-2) +\
			   (torch.unsqueeze(ray_direction, dim=-2) * torch.unsqueeze(z_vals, dim=-1))
		rays   = torch.reshape(rays, (self.batch_size, -1, self.num_samples, 3))
		rays   = torch.squeeze(self.encode_position(rays, self.pos_enc_dim), dim=0)

		z_vals = torch.tile(torch.unsqueeze(z_vals, dim=0), (rays.shape[0], 1))

		return (rays, z_vals)


	# Source: https://github.com/bmild/nerf/blob/20a91e764a28816ee2234fcadb73bd59a613a44c/run_nerf_helpers.py#L183
	def inverse_transform_sampling(self, t_vals_mid, weights):

		# t_vals_mid -> B x H x W x (num_samples-1) -> B x H x W x (num_samples)
		t_vals_mid = torch.concat([t_vals_mid, torch.ones(size=(t_vals_mid.shape[0], 1)).to(self.device)*(torch.max(t_vals_mid)+1e-5)], dim=-1)

		# Adding a epsilon weight to prevent from NaN
		weights_c = weights + 1e-5

		# Normalize weights to get PDF
		pdf = weights_c / torch.sum(weights_c, dim=-1, keepdim=True)

		# Computing CDF
		# B x H x W x num_samples
		cdf = torch.cumsum(pdf, dim=-1)
		# cdf = cdf.reshape(self.batch_size, -1, cdf.shape[-1])

		# Adding zero at the beginning of CDF
		# B x H x W x (num_samples + 1)
		# cdf = torch.concat([torch.zeros(size=(self.batch_size, self.height, self.width, 1)).to(self.device), cdf], dim=-1)

		# Inverse transform uniform dist -> required PDF
		# Searchsorted will give indices which can be interprted as 
		# which t_vals helps in generating the density
		uniform_sample = torch.rand(size=(pdf.shape[0], self.num_samples, 1)).to(self.device)

		indices        = torch.searchsorted(cdf, uniform_sample, side='right')
		
		# Boundaries, logic not clear
		below = torch.maximum(torch.zeros_like(indices).to(self.device), indices-1)
		above = torch.minimum(torch.ones_like(indices).to(self.device)* cdf.shape[-1]-1, indices)
		# indices_stack = torch.stack([below, above], dim=-1)

		# Accumulating CDF according to the bound
		# cdf_stack = self.gather_cdf_util(cdf, indices_stack)
		cdf_gather_lower = torch.gather(input=cdf, dim=-1, index=below)
		cdf_gather_above = torch.gather(input=cdf, dim=-1, index=above)
		
		# Accumulating t_vals_mid according to the bound
		# cdf_stack = self.gather_cdf_util(cdf, indices_stack)
		t_vals_mid_gather_below = torch.gather(input=torch.unsqueeze(t_vals_mid, dim=-1), dim=-1, index=below)
		t_vals_mid_gather_above = torch.gather(input=torch.unsqueeze(t_vals_mid, dim=-1), dim=-1, index=above)

		# Creating sampling points
		denom = cdf_gather_above - cdf_gather_lower
		denom = torch.where(denom < 1e-5, torch.ones_like(denom).to(self.device), denom)
		t = (uniform_sample - cdf_gather_lower)/denom
		t_vals_fine = t_vals_mid_gather_below + t * (t_vals_mid_gather_above - t_vals_mid_gather_below)

		return torch.squeeze(t_vals_fine, dim=-1)
	
	def sampling_fine_rays(self, ray_origin, ray_direction, t_vals, weights):
		# Finding mid values for t_vals
		t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1]) # mid = [(a+b)/2]

		with torch.no_grad():
			# Finding finer t_vals
			t_vals_fine = self.inverse_transform_sampling(t_vals_mid, weights)

			# Merging coarse t_vals and fine t_vals
			t_vals_fine, _ = torch.sort(torch.concat([t_vals, t_vals_fine], dim=-1), dim=-1)

		# r(t) = o + td -> building "r" here, using fine t_vals
		# B x H x W x 1 x 3 + B x H x W x 1 x 3 * B x H x W x 32 x 1
		rays = torch.unsqueeze(ray_origin, dim=-2) +\
			   (torch.unsqueeze(ray_direction, dim=-2) * torch.unsqueeze(t_vals_fine, dim=-1))
		# rays = torch.reshape(rays, (self.batch_size, -1, 32, 3))
		rays = torch.squeeze(self.encode_position(rays, self.pos_enc_dim), dim=0)

		return (rays, t_vals_fine)


	def render_rgb_depth(self, prediction, rays, t_vals, random_sampling=True):
		
		# Slice the prediction
		rgb     = torch.nn.Sigmoid()(prediction[..., :-1])
		density = torch.nn.Sigmoid()(prediction[..., -1:])

		# pdb.set_trace()
		# print('imin: {}, imax: {}'.format(rgb.min(), rgb.max()))
		# print('dmin: {}, dmax: {}'.format(density.min(), density.max()))

		# computing delta
		# output will be one less dimension
		delta = t_vals[..., 1:] - t_vals[..., :-1]

		if random_sampling:

			# padding dimension
			# B x H x W x num_samples-1 -> B x H x W x num_samples

			delta = torch.concat([
									delta,
									torch.ones(size=(delta.shape[0], 1)).to(self.device) * 1e10
								], dim=-1)

			delta = torch.unsqueeze(delta, dim=-1)

		alpha = 1.0 - torch.exp(-density * delta)

		# print('alphamin: {}, alphamax: {}'.format(alpha.min(), alpha.max()))

		# calculating transmittance
		# exp_term = 1.0 - alpha
		epsilon       = 1e-10
		transmittance = torch.exp(-torch.cumsum(density * delta, dim=-1) + epsilon)
		# transmittance = torch.cumprod(exp_term + epsilon, dim=-1)
		weights  = alpha * transmittance

		# Accumulating radiance along the rays
		# B x H x W x num_sample x 1, B x H x W x 32 x 3
		rgb = torch.sum( weights * rgb, axis=-2)

		# calculating depth map using density and sample points
		if random_sampling:
			depth_map = torch.sum( torch.squeeze(weights, dim=-1) * t_vals, axis=-1)

		# print('imin: {}, imax: {}'.format(rgb.min(), rgb.max()))
		# print('dmin: {}, dmax: {}'.format(depth_map.min(), depth_map.max()))

		return rgb, depth_map, weights

	# def sub_batching(self, rays, t_vals, image, chunk_size=4096):
	# 	rays_sub   = [rays[i:i+chunk_size] for i in range(0, rays.shape[0], chunk_size)]
	# 	t_vals_sub = [t_vals[i:i+chunk_size] for i in range(0, t_vals.shape[0], chunk_size)]
	# 	image_sub  = [image[i:i+chunk_size] for i in range(0, rays.shape[0], chunk_size)]
		
	# 	return rays_sub, t_vals_sub, image_sub
	def sub_batching(self, rays, t_vals, chunk_size=4096):
		rays_sub   = [rays[i:i+chunk_size] for i in range(0, rays.shape[0], chunk_size)]
		t_vals_sub = [t_vals[i:i+chunk_size] for i in range(0, t_vals.shape[0], chunk_size)]		
		return rays_sub, t_vals_sub


if __name__ == '__main__':

	pos  = torch.randn(size=(config.batch_size, 3)).to(config.device)
	dirc = torch.randn(size=(config.batch_size, 3)).to(config.device)
	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   focal=177.77,\
							   batch_size=config.batch_size,\
							   near=config.near_plane,\
							   far=config.far_plane,\
							   num_samples=config.num_samples,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	# x_pos = nerf_comp.encode_position(x=pos, enc_dim=config.pos_enc_dim)
	# x_dir = nerf_comp.encode_position(x=pos, enc_dim=config.pos_enc_dim)
	# print(pos.shape, x_pos.shape, x_dir.shape)

	# ray_origin, ray_direction = nerf_comp.get_rays(camera_matrix=torch.rand(size=(config.batch_size, 4, 4)))
	# print(ray_origin.shape, ray_direction.shape)

	rays, t_vals = nerf_comp.sampling_rays(camera_matrix=torch.rand(size=(config.batch_size, 4, 4)).to(config.device), random_sampling=True)
	rays_sub, t_vals_sub = nerf_comp.sub_batching(rays, t_vals)
	# print(rays_sub[0].shape, t_vals_sub[0].shape)
	# print(rays.shape, t_vals.shape)
	
	rgb, depth_map, weights = nerf_comp.render_rgb_depth(torch.randn(size=(config.batch_size, config.image_height, config.image_width, config.num_samples, 4)).to(config.device), rays, t_vals)
	# print(rgb.shape, depth_map.shape, weights.shape)

	fine_rays, t_vals_fine = nerf_comp.sampling_fine_rays(torch.rand(size=(config.batch_size, 4, 4)).to(config.device), t_vals, weights)

	print(rays.shape, t_vals.shape, rgb.shape, fine_rays.shape, t_vals_fine.shape)