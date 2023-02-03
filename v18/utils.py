import os
import matplotlib.pyplot as plt
from matplotlib import style
from torchvision import transforms
import numpy as np
import torch
import config

plt.rcParams["savefig.bbox"] = 'tight'
# plt.rcParams["figure.figsize"] = (18,18)
style.use('seaborn')
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
def show(imgs, path, label, idx):
    path = '{}/{}/'.format(path, label)
    if not os.path.isdir(path):
        os.makedirs(path)

    # if not isinstance(imgs, list):
    #     imgs = imgs[:]

    # fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    # for i, img in enumerate(imgs):
    #     img = img.detach()
    #     img = transforms.functional.to_pil_image(img)
    #     img = np.asarray(img)
    #     if len(img.shape) < 3:
    #         axs[0, i].imshow(img, cmap='viridis')
    #     else:
    #         axs[0, i].imshow(img)
    #     axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(11, 11), dpi=96)
    plt.figure(figsize=(9, 9), dpi=96)

    if len(imgs.shape)<3:
        plt.imshow(imgs.cpu().detach().numpy(), cmap='viridis')
        # plt.imshow(imgs.cpu().detach().numpy(), cmap='gray')
    else:
        plt.imshow(imgs.cpu().detach().numpy())
    plt.axis('off')
    plt.grid(False)
    plt.savefig('{}/{}.png'.format(path, idx))
    plt.close()

def mse2psnr(x):
	return -10.*torch.log(x)/torch.log(torch.as_tensor(10., dtype=torch.float32))
