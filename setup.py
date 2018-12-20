import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('setup')

dataset_folder = 'dataset/'

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

os.chdir('dataset')
os.system('python3 parse_datasets.py')
logger.info('done')
