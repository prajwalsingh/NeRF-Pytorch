import os
import math

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

dataset_type    = 'synthetic' # 'synthetic', 'real', 'llff'

if dataset_type == 'synthetic':
	train_camera_path = 'dataset/nerf_synthetic/lego/transforms_train.json' 
	val_camera_path   = 'dataset/nerf_synthetic/lego/transforms_val.json' 
	train_image_path  = 'dataset/nerf_synthetic/lego'
	val_image_path    = 'dataset/nerf_synthetic/lego'

elif (dataset_type == 'real') or (dataset_type == 'llff'):
	train_camera_path = 'dataset/nerf_real_360/vasedeck/poses_bounds.npy'
	val_camera_path   = 'dataset/nerf_real_360/vasedeck/poses_bounds.npy'
	train_image_path  = 'dataset/nerf_real_360/vasedeck/images'
	val_image_path    = 'dataset/nerf_real_360/vasedeck/images'

	# train_camera_path = 'dataset/nerf_llff_data/fern/poses_bounds.npy'
	# val_camera_path   = 'dataset/nerf_llff_data/fern/poses_bounds.npy'
	# train_image_path  = 'dataset/nerf_llff_data/fern/images'
	# val_image_path    = 'dataset/nerf_llff_data/fern/images'

device       = 'cuda'
use_ndc      = False
lr           = 5e-4
image_height = 512
image_width  = 512
num_channels = 3
batch_size   = 1
epochs       = 5001#10001
vis_freq     = 10
ckpt_freq    = 10
lrsch_step   = 2500#20
lrsch_gamma  = 0.1#0.99

pos_enc_dim  = 10
dir_enc_dim  = 4

pre_epoch    = 50
pre_crop     = 0.5
noise_value  = 0.0
near_plane   = 2.0
far_plane    = 6.0
num_samples  = 64
num_samples_fine  = 128
net_dim      = 256
in_feat      = 2*(3*pos_enc_dim) + 3
dir_feat     = 2*(3*dir_enc_dim) + 3
skip_layer   = 4
net_depth    = 8
num_samples_fine  = 2 * num_samples
n_samples    = 1024#4096