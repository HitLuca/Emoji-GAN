import sys
import numpy as np
from wgan_gp import WGAN_GP
sys.path.append("..")
import utils

batch_size = 32
resolution = 16
channels = 4

dataset = np.load('../../dataset/emojis_' + str(resolution) + '.npy')
perm = np.random.permutation(dataset.shape[0])
dataset = dataset[perm]

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'batch_size': batch_size,
    'channels': channels,
    'resolution': resolution,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
}

config = utils.merge_config_and_save(config_2)

wgan_gp = WGAN_GP(config)
losses = wgan_gp.train(dataset)