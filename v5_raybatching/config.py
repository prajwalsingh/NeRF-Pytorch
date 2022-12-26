import os
import math

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

train_json_path = 'dataset/nerf_synthetic/hotdog/transforms_train.json' 
val_json_path   = 'dataset/nerf_synthetic/hotdog/transforms_val.json' 
image_path      = 'dataset/nerf_synthetic/hotdog'
device       = 'cuda'
lr           = 5e-4
image_height = 256
image_width  = 256
batch_size   = 1
epochs       = 500

pos_enc_dim  = 10
dir_enc_dim  = 4

near_plane   = 2.0
far_plane    = 6.0
num_samples  = 64
net_dim      = 256
in_feat      = 63
skip_layer   = 4
net_depth    = 8
num_samples_fine  = 2 * num_samples
chunk_size   = 4096 