import torch
import numpy as np
import config

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
		
		positions = [x]

		for i in range(enc_dim):
			positions.append(torch.sin( (2.0**i) * torch.pi * x ))
			positions.append(torch.cos( (2.0**i) * torch.pi * x ))

		return torch.concat(positions, axis=-1).to(self.device)

	def get_rays(self, camera_matrix):

		# Rotation matrix, camera to world coordinates C_ext^{-1}
		rotation_matrix = camera_matrix[:, :3, :3].to(self.device)
		# Translation matrix from camera to world coordinates t_{ext}^{-1}
		translation     = camera_matrix[:, :3, -1].to(self.device)

		# Applying rotation matrix in left side, following is method to do so
		# without doing transpose
		# camera to world coordinates
		self.direction_c   = torch.unsqueeze(self.direction, dim=-2)
		rotation_matrix    = torch.unsqueeze(torch.unsqueeze(rotation_matrix, dim=1), dim=1)
		ray_direction      = torch.sum(self.direction_c * rotation_matrix, dim=-1)

		# Ray origin
		ray_origin         = torch.tile(torch.unsqueeze(torch.unsqueeze(translation, dim=-2), dim=-2), (1, self.height, self.width, 1))

		return (ray_origin, ray_direction)

	def sampling_rays(self, camera_matrix, random_sampling=True):

		ray_origin, ray_direction = self.get_rays(camera_matrix)

		# Compute 3D query points
		# r(t) = o + td -> we are buildin t here [x,y,z] of raidance
		# this is discrete sampling
		t_vals = torch.linspace(self.near, self.far, self.num_samples).to(self.device)

		# continuos sampling
		if random_sampling:
			 # Injecting uniform noise to make sampling continuos
			 # B x H x W x num_samples
			 shape  = list(ray_origin.shape[:-1]) + [self.num_samples]
			 noise  = torch.rand(size=shape).to(self.device) * ((self.far - self.near) / self.num_samples)
			 t_vals = t_vals + noise

		# r(t) = o + td -> building "r" here
		# B x H x W x 1 x 3 + B x H x W x 1 x 3 * B x H x W x 32 x 1
		rays = torch.unsqueeze(ray_origin, dim=-2) +\
			   (torch.unsqueeze(ray_direction, dim=-2) * torch.unsqueeze(t_vals, dim=-1))
		rays = torch.reshape(rays, (self.batch_size, -1, 3))
		rays = self.encode_position(rays, self.pos_enc_dim)

		return (rays, t_vals)

	def render_rgb_depth(self, prediction, rays, t_vals, random_sampling=True):
		
		# Slice the prediction
		rgb     = torch.nn.Sigmoid()(prediction[..., :-1])
		density = torch.nn.Sigmoid()(prediction[..., -1])

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
									torch.ones(size=(self.batch_size, self.height, self.width, 1)).to(self.device) * 1e10
								], dim=-1)

		alpha = 1.0 - torch.exp(-density * delta)

		# print('alphamin: {}, alphamax: {}'.format(alpha.min(), alpha.max()))

		# calculating transmittance
		exp_term = 1.0 - alpha
		epsilon  = 1e-10
		transmittance = torch.cumprod(exp_term + epsilon, axis=-1)
		weights  = alpha * transmittance

		# Accumulating radiance along the rays
		# B x H x W x num_sample x 1, B x H x W x 1 x 3
		rgb = torch.sum( torch.unsqueeze(weights, dim=-1) * rgb, axis=-2)

		# calculating depth map using density and sample points
		if random_sampling:
			depth_map = torch.sum(weights * t_vals, axis=-1)

		# print('imin: {}, imax: {}'.format(rgb.min(), rgb.max()))
		# print('dmin: {}, dmax: {}'.format(depth_map.min(), depth_map.max()))

		return rgb, depth_map


if __name__ == '__main__':

	pos  = torch.randn(size=(config.batch_size, 3))
	dirc = torch.randn(size=(config.batch_size, 3))
	nerf_comp = NerfComponents(height=config.image_height,\
							   width=config.image_width,\
							   focal=config.focal,\
							   batch_size=config.batch_size,\
							   near=config.near_plane,\
							   far=config.far_plane,\
							   num_samples=config.num_samples,\
							   pos_enc_dim=config.pos_enc_dim,\
							   dir_enc_dim=config.dir_enc_dim)

	x_pos = nerf_comp.encode_position(x=pos, enc_dim=config.pos_enc_dim)
	x_dir = nerf_comp.encode_position(x=pos, enc_dim=config.pos_enc_dim)
	# print(pos.shape, x_pos.shape, x_dir.shape)

	ray_origin, ray_direction = nerf_comp.get_rays(camera_matrix=torch.rand(size=(config.batch_size, 4, 4)))
	# print(ray_origin.shape, ray_direction.shape)

	nerf_comp.sampling_rays(camera_matrix=torch.rand(size=(config.batch_size, 4, 4)), random_sampling=True)