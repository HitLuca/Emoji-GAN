import logging
import os
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('setup')

dataset_folder = 'dataset/'
img_folder = dataset_folder + 'img/'

emoji_repo_url = 'https://github.com/iamcal/emoji-data/'
img_repo_url = emoji_repo_url + 'raw/master/'

categories_json = 'categories.json'
categories_repo_url = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/' + categories_json

emoji_json = 'emoji.json'
emoji_repo_url = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/' + emoji_json

companies = ['apple', 'facebook', 'google', 'messenger', 'twitter']
resolutions = [16, 20, 32, 64]

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

logger.info('downloading dataset...')
for company in companies:
    for resolution in resolutions:
        filename = '_'.join(['sheet', company, str(resolution)]) + '.png'
        filepath = img_folder + filename
        if not os.path.exists(filepath):
            logger.info(filename + ' not found, downloading...')
            urllib.request.urlretrieve(img_repo_url + filename, filepath)

if not os.path.exists(dataset_folder + categories_json):
    logger.info(categories_json + ' not found, downloading...')
    urllib.request.urlretrieve(categories_repo_url, dataset_folder + categories_json)

if not os.path.exists(dataset_folder + emoji_json):
    logger.info(emoji_json + ' not found, downloading...')
    urllib.request.urlretrieve(emoji_repo_url, dataset_folder + emoji_json)

logger.info('done')

logger.info('converting and parsing dataset')
os.chdir('dataset')
os.system('python3 parse_datasets.py')
logger.info('done')
