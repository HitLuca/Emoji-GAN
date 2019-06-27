import json
import logging
import os
import urllib.request

import imageio
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('setup_datasets')


img_folder = 'dataset/img/'

emoji_repo_url = 'https://github.com/iamcal/emoji-data/'
img_repo_url = emoji_repo_url + 'raw/master/'

categories_json = 'categories.json'
categories_repo_url = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/' + categories_json

emoji_json = 'emoji.json'
emoji_repo_url = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/' + emoji_json

companies = ['apple', 'facebook', 'google', 'messenger', 'twitter']
resolutions = [16, 20, 32, 64]


def download_dataset(img_folder):
    logger.info('downloading dataset...')
    for company in companies:
        for resolution in resolutions:
            filename = '_'.join(['sheet', company, str(resolution)]) + '.png'
            filepath = img_folder + filename
            if not os.path.exists(filepath):
                logger.info(filename + ' not found, downloading...')
                urllib.request.urlretrieve(img_repo_url + filename, filepath)

    if not os.path.exists(categories_json):
        logger.info(categories_json + ' not found, downloading...')
        urllib.request.urlretrieve(categories_repo_url, categories_json)

    if not os.path.exists(emoji_json):
        logger.info(emoji_json + ' not found, downloading...')
        urllib.request.urlretrieve(emoji_repo_url, emoji_json)

    logger.info('done')


def png_to_dataset(png, resolution):
    emojis = []
    height, width, _ = png.shape
    for i in range(1, height, resolution + 2):
        for j in range(1, width, resolution + 2):
            emoji = png[j:j + resolution, i:i + resolution]
            if np.max(emoji[:, :, -1] != 0):
                emojis.append(emoji)
    return np.array(emojis)


if __name__ == "__main__":
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
        download_dataset(img_folder)

    categories_data = json.load(open('categories.json', 'r'))
    categories_names = list(categories_data.keys())
    emoji_data = json.load(open('emoji.json', 'r'))

    with open('companies_names.json', 'w') as f:
        json.dump(companies, f)

    with open('categories_names.json', 'w') as f:
        json.dump(categories_names, f)

    logger.info('generating datasets...')
    for resolution in resolutions:
        dataset = []
        classes = []

        companies_pngs = []
        for company in companies:
            sheet_filepath = img_folder + ('_'.join(['sheet', company, str(resolution)])) + '.png'
            companies_pngs.append(imageio.imread(sheet_filepath))

        for element in emoji_data:
            valid = True
            for company in companies:
                if not element['has_img_' + company]:
                    valid = False
                    break
            if valid:
                y = element['sheet_x']
                x = element['sheet_y']
                category = element['category']

                base_x = 1 + x * (2 + resolution)
                base_y = 1 + y * (2 + resolution)

                for company in companies:
                    company_index = companies.index(company)
                    category_index = categories_names.index(category)

                    emoji = companies_pngs[company_index][base_x:base_x + resolution, base_y:base_y + resolution, :]
                    dataset.append(emoji)
                    classes.append([company_index, category_index])
        dataset = np.array(dataset)
        dataset = dataset / 255.0

        classes = np.array(classes)

        np.save('emojis_' + str(resolution) + '.npy', dataset.astype(np.float32))
        np.save('emojis_classes' + '.npy', classes)

    logger.info('done')
