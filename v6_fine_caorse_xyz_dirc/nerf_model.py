import torch
import config
import torch.nn as nn

class NerfNet(nn.Module):

	def __init__(self, depth, xyz_feat, dirc_feat, net_dim=128, skip_layer=4):
		super(NerfNet, self).__init__()
		self.depth = depth
		self.skip_layer = skip_layer
		units = [xyz_feat] + [net_dim]*(self.depth+1)
		self.layers = nn.ModuleList([])
		self.bnorm_layers = nn.ModuleList([])

		# self.act    = nn.ReLU()
		# self.act    = nn.SiLU()
		self.act     = nn.GELU()
		self.act_out = nn.Sigmoid()

		for i in range(self.depth):
			if (i%self.skip_layer==0) and (i>0):
				self.layers.append(nn.Linear(in_features=units[i]+xyz_feat, out_features=units[i+1]))
				# self.bnorm_layers.append(nn.InstanceNorm1d(num_features=units[i+1]))
			else:	
				self.layers.append(nn.Linear(in_features=units[i], out_features=units[i+1]))
				# self.bnorm_layers.append(nn.InstanceNorm1d(num_features=units[i+1]))

		self.layers.append(nn.Linear(in_features=net_dim, out_features=1))
		self.layers.append(nn.Linear(in_features=1 + dirc_feat, out_features=net_dim))
		self.layers.append(nn.Linear(in_features=net_dim, out_features=net_dim))
		self.layers.append(nn.Linear(in_features=net_dim, out_features=3))

	def forward(self, xyz, dirc):
		
		x = xyz

		for i in range(self.depth):

			if (i%self.skip_layer==0) and (i>0):
				x = torch.concat([x, xyz], dim=-1)

			# x = self.act(self.bnorm_layers[i]( self.layers[i]( x )) )
			x = self.act( self.layers[i]( x ) )

		density = self.layers[self.depth]( x )
		rgb     = self.act( self.layers[self.depth+1]( torch.concat([density, dirc], dim=-1) ) )
		rgb     = self.act( self.layers[self.depth+2]( rgb ) )
		rgb     = self.layers[self.depth+3]( rgb )
		# print('omin: {}, omax: {}'.format(out.min(), out.max()))

		return self.act_out( density ), self.act_out( rgb )
		


if __name__ == '__main__':
	device = 'cuda'
	inp = torch.rand(size=[5, 524288, 63]).to(device)
	nerfnet = NerfNet(depth=config.net_depth, in_feat=config.in_feat,\
					  net_dim=config.net_dim, skip_layer=config.skip_layer).to(device)

	out = nerfnet(inp)
	print(out.shape)
	out = torch.reshape(out, (config.batch_size, config.image_height, config.image_width, config.num_samples, out.shape[-1]))
	print(out.shape)