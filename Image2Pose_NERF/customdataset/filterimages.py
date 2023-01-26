import os
import shutil
from tqdm import tqdm
from glob import glob
from natsort import natsorted

if __name__ == '__main__':

	if not os.path.isdir('images'):
		os.makedirs('images')

	paths    = natsorted(glob('extract/*'))
	skip_idx = 8
	for path in tqdm(paths[::skip_idx]):
		shutil.copy(path, 'images/')