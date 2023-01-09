import os
import math

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

dataset_type    = 'json' # 'json' or 'npy'

train_json_path = 'dataset/nerf_synthetic/hotdog/transforms_train.json' 
val_json_path   = 'dataset/nerf_synthetic/hotdog/transforms_val.json' 
image_path      = 'dataset/nerf_synthetic/hotdog'

train_camera_path = 'dataset/nerf_real_360/vasedeck/poses_bounds.npy'
val_camera_path   = 'dataset/nerf_real_360/vasedeck/poses_bounds.npy'
train_npy_path    = 'dataset/nerf_real_360/vasedeck/images'
val_npy_path      = 'dataset/nerf_real_360/vasedeck/images'

device       = 'cuda'
lr           = 5e-4
image_height = 512
image_width  = 512
batch_size   = 1
epochs       = 2001
vis_freq     = 10
lrsch_step   = 20
lrsch_gamma  = 0.99

pos_enc_dim  = 16
dir_enc_dim  = 12

near_plane   = 2.0
far_plane    = 6.0
num_samples  = 64
net_dim      = 256
in_feat      = 2*(3*pos_enc_dim) + 3
dir_feat     = 2*(3*dir_enc_dim) + 3
# in_feat      = 60
skip_layer   = 4
net_depth    = 8
num_samples_fine  = 2 * num_samples
n_samples    = 2048#4096