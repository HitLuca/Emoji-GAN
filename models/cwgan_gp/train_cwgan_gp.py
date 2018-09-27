import sys
import numpy as np
from cwgan_gp import CWGAN_GP
sys.path.append("..")
import utils

batch_size = 32
resolution = 16
channels = 4

companies = np.load('../../dataset/companies_names.npy')
categories_names = np.load('../../dataset/categories_names.npy')

dataset = np.load('../../dataset/emojis_' + str(resolution) + '.npy')
classes = np.load('../../dataset/emojis_classes.npy')[:, 1]
classes_n = categories_names.shape[0]

perm = np.random.permutation(dataset.shape[0])
dataset = dataset[perm]
classes = classes[perm]

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'batch_size': batch_size,
    'classes_n': classes_n,
    'class_names': categories_names.tolist(),
    'channels': channels,
    'resolution': resolution,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
}

config = utils.merge_config_and_save(config_2)

cwgan_gp = CWGAN_GP(config)
losses = cwgan_gp.train(dataset, classes)
