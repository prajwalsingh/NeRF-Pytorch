import random
import torch
import numpy as np
import config
import torch.nn as nn

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
		


if __name__ == '__main__':
	device = 'cuda'
	inp = torch.rand(size=[5, 524288, 63]).to(device)
	nerfnet = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(device)

	out = nerfnet(inp)
	print(out.shape)
	out = torch.reshape(out, (config.batch_size, config.image_height, config.image_width, config.num_samples, out.shape[-1]))
	print(out.shape)