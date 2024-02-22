import random
import torch
import numpy as np
import pdb
import nerfacc

torch.manual_seed(45)
np.random.seed(45)
random.seed(45)

class NerfComponents:
	def __init__(self, height, width, batch_size, num_samples_coarse, num_samples_fine, pos_enc_dim, dir_enc_dim, args=None):
		self.height = height
		self.width  = width
		self.batch_size = batch_size
		self.num_samples = num_samples_coarse
		self.num_samples_fine = num_samples_fine
		self.pos_enc_dim = pos_enc_dim
		self.dir_enc_dim = dir_enc_dim
		self.args = args
		aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
		self.estimator   = nerfacc.OccGridEstimator(roi_aabb=aabb)

	def encode_position(self, x, enc_dim):
		
		positions = [x]
		# positions = []

		for i in range(enc_dim):
			# positions.append(torch.sin( (2.0**i) * torch.pi * x ))
			# positions.append(torch.cos( (2.0**i) * torch.pi * x ))
			positions.append(torch.sin( (2.0**i) *  x ))
			positions.append(torch.cos( (2.0**i) *  x ))

		return torch.concat(positions, dim=-1)
	

	def get_rays(self, camera_matrix, direction):

		# Rotation matrix, camera to world coordinates C_ext^{-1}
		rotation_matrix = camera_matrix[:3, :3]
		# Translation matrix from camera to world coordinates t_{ext}^{-1}
		translation     = camera_matrix[:3, -1]

		# Applying rotation matrix in left side, following is method to do so
		# without doing transpose
		# camera to world coordinates
		direction_c        = torch.unsqueeze(direction, dim=-2)
		rotation_matrix_c  = torch.unsqueeze(rotation_matrix, dim=0)
		ray_direction      = torch.sum(direction_c * rotation_matrix_c, dim=-1)

		# Normalizing the direction vector
		# ray_direction      = ray_direction / torch.unsqueeze(torch.norm(ray_direction, dim=-1), dim=-1)

		# Ray origin
		ray_origin         = torch.tile(torch.unsqueeze(translation, dim=0), (ray_direction.shape[0], 1))

		return (ray_origin, ray_direction)
	
	
	def sampling_rays(self, ray_origin, ray_direction, near, far, random_sampling=True, device=None):

		# Compute 3D query points
		# r(t) = o + td -> we are buildin t here [x,y,z] of raidance
		# this is discrete sampling
		t_vals = torch.linspace(0.0, 1.0, self.num_samples, device=device) # N_samples
		
		# Inverse Depth Sampling
		# t_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # N_samples

		# Depth Sampling
		t_vals = near * (1.-t_vals) + far * (t_vals)
		
		t_vals = torch.tile(torch.unsqueeze(t_vals, dim=0), [ray_direction.shape[0], 1]) # N_rays x N_samples

		# continuos sampling
		if random_sampling:
			# Injecting uniform noise to make sampling continuos
			# B x H x W x num_samples
			# shape  = list(ray_origin.shape[:-1]) + [self.num_samples]
			# noise  = torch.rand(size=shape) * ((self.far - self.near) / self.num_samples)
			# t_vals = t_vals + noise
			mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1]) # N_rays x (N_samples-1)
			upper = torch.concat([mids, t_vals[..., -1:]], dim=-1) # N_rays x N_samples
			lower = torch.concat([t_vals[..., :1], mids], dim=-1) # N_rays x N_samples
			# stratified samples in those intervals
			t_rand = torch.rand(t_vals.shape, device=device) # N_rays x N_samples
			t_vals = lower + (upper - lower) * t_rand # N_rays x N_samples

		# r(t) = o + td -> building "r" here
		# B x H x W x 1 x 3 + B x H x W x 1 x 3 * B x H x W x 32 x 1
		# N_rays x 3 = N_rays x 1 x 3 + N_rays x 1 x 3 + N_rays x N_samples x 1
		rays = torch.unsqueeze(ray_origin, dim=-2) +\
			   (torch.unsqueeze(ray_direction, dim=-2) * torch.unsqueeze(t_vals, dim=-1))

		# rays   = torch.reshape(rays, (self.batch_size, -1, self.num_samples, 3))

		# rays   = torch.squeeze(self.encode_position(rays, self.pos_enc_dim), dim=0)
		rays   = self.encode_position(rays, self.pos_enc_dim)

		# t_vals = torch.tile(torch.unsqueeze(z_vals, dim=0), (rays.shape[0], 1))
		# t_vals = torch.squeeze(z_vals, dim=0)

		return (rays, t_vals)


	# Source: https://github.com/bmild/nerf/blob/20a91e764a28816ee2234fcadb73bd59a613a44c/run_nerf_helpers.py#L183
	def inverse_transform_sampling(self, t_vals_mid, weights, device):

		# t_vals_mid -> B x H x W x (num_samples-1) -> B x H x W x (num_samples)
		# t_vals_mid = torch.concat([t_vals_mid, torch.ones(size=(t_vals_mid.shape[0], 1))*(torch.max(t_vals_mid)+1e-5)], dim=-1)
		# t_vals_mid = torch.concat([torch.zeros(size=(t_vals_mid.shape[0], 1)), t_vals_mid], dim=-1)

		# Adding a epsilon weight to prevent from NaN
		weights_c = weights + 1e-5 # N_rays x N_samples

		# Normalize weights to get PDF
		pdf = weights_c / torch.sum(weights_c, dim=-1, keepdim=True) # N_rays x N_samples

		# Computing CDF
		cdf = torch.cumsum(pdf, dim=-1) # N_rays x N_samples

		# Adding zero at the beginning of CDF
		cdf = torch.concat([torch.zeros_like(weights_c[..., :1], device=device), cdf], dim=-1) # N_rays x ( num_samples + 1 )

		# Inverse transform uniform dist -> required PDF
		# Searchsorted will give indices which can be interprted as 
		# which t_vals helps in generating the density
		uniform_sample = torch.rand(size=(cdf.shape[0], self.num_samples_fine), device=device) # N_rays x N_sample_fine

		indices        = torch.searchsorted(cdf, uniform_sample, side='right')
		
		# Boundaries, logic not clear
		below = torch.maximum(torch.zeros_like(indices, device=device), indices-3)
		above = torch.minimum(torch.ones_like(indices, device=device)* (cdf.shape[-1]-3), indices)
		# indices_stack = torch.stack([below, above], dim=-1)

		# Accumulating CDF according to the bound
		# cdf_stack = self.gather_cdf_util(cdf, indices_stack)
		cdf_gather_lower = torch.gather(input=cdf, dim=-1, index=below)
		cdf_gather_above = torch.gather(input=cdf, dim=-1, index=above)

		# Accumulating t_vals_mid according to the bound
		# cdf_stack = self.gather_cdf_util(cdf, indices_stack)
		t_vals_mid_gather_below = torch.gather(input=t_vals_mid, dim=-1, index=below)
		t_vals_mid_gather_above = torch.gather(input=t_vals_mid, dim=-1, index=above)

		# Creating sampling points
		denom = cdf_gather_above - cdf_gather_lower
		denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
		t = (uniform_sample - cdf_gather_lower)/denom
		t_vals_fine = t_vals_mid_gather_below + t * (t_vals_mid_gather_above - t_vals_mid_gather_below)

		return torch.squeeze(t_vals_fine, dim=-1)
	
	def sampling_fine_rays(self, ray_origin, ray_direction, t_vals, weights, device):
		# Finding mid values for t_vals
		t_vals_mid = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1]) # mid = [(a+b)/2]

		# Finding finer t_vals
		t_vals_fine = self.inverse_transform_sampling(t_vals_mid, weights, device)
		t_vals_fine = t_vals_fine.detach()

		# Merging coarse t_vals and fine t_vals
		t_vals_fine, _ = torch.sort(torch.concat([t_vals, t_vals_fine], dim=-1), dim=-1)

		# r(t) = o + td -> building "r" here, using fine t_vals
		# B x H x W x 1 x 3 + B x H x W x 1 x 3 * B x H x W x 32 x 1
		rays = torch.unsqueeze(ray_origin, dim=-2) +\
			   (torch.unsqueeze(ray_direction, dim=-2) * torch.unsqueeze(t_vals_fine, dim=-1))
		# rays = torch.reshape(rays, (self.batch_size, -1, 32, 3))
		# rays = torch.squeeze(self.encode_position(rays, self.pos_enc_dim), dim=0)
		rays = self.encode_position(rays, self.pos_enc_dim)

		return (rays, t_vals_fine)


	def render_rgb_depth(self, rgb, density, rays_d, t_vals, noise_value=0.0, random_sampling=True, device=None):
		
		rgb     = torch.nn.Sigmoid()(rgb) # N_rays x N_samples x 3
		density = torch.squeeze(density, dim=-1) # N_rays x N_samples x 1 -> N_rays x N_samples
		
		noise = 0.0
		if noise_value>0.0:
			noise = torch.normal(mean=0.0, std=1.0, size=(density.shape), device=device) # N_rays x N_samples
		# noise = torch.normal(mean=0.0, std=1.0, size=(density.shape)) # N_rays x N_samples

		# As per NeRF paper
		density = torch.nn.ReLU()(density+noise) # N_rays x N_samples

		# As per Nerfies paper
		# density = torch.nn.Softplus()(density+noise) # N_rays x N_samples

		delta = t_vals[..., 1:] - t_vals[..., :-1] # N_rays x (N_samples-1)

		# padding dimension
		# N_rays x (N_samples-1) -> N_rays x N_samples
		delta = torch.concat([
								delta,
								torch.ones(size=(delta.shape[0], 1), device=device) * 1e10
							], dim=-1)

		# delta: N_rays x N_samples
		# rays_d: B x N_rays x 3 -> N_rays x 1 x 3 -> 2-norm -> N_rays x 1
		# N_rays x N_samples

		delta = delta * torch.norm(rays_d, dim=-1, keepdim=True) # Euclidean Norm

		# N_rays x N_samples = N_rays x N_samples * N_rays x N_samples
		exp_term = torch.exp(-density * delta)
		alpha    = 1.0 - exp_term

		# calculating transmittance
		epsilon  = 1e-10
		# transmittance = torch.exp(-torch.cumsum(density * delta, dim=-1) + epsilon)
		transmittance = torch.cumprod(exp_term + epsilon, dim=-1)

		# Based on: https://github.com/sillsill777/NeRF-PyTorch
		transmittance = torch.cat([torch.ones_like(transmittance[..., :1], device=device), transmittance], dim=-1)[..., :-1]
		# Why removed last value?

		weights  = alpha * transmittance

		# Accumulating radiance along the rays
		# N_rays x N_samples x 3 = N_rays x N_samples x 1 * N_rays x N_samples x 3
		rgb = torch.sum( weights[..., None] * rgb, axis=-2)

		# calculating depth map using density and sample points
		# N_rays x N_samples = N_rays x N_samples * N_rays x N_samples
		depth_map = torch.sum( weights * t_vals, axis=-1)

		return rgb, depth_map, weights