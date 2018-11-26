import json
import logging

import imageio
import numpy as np


def png_to_dataset(png, resolution):
    emojis = []
    height, width, _ = png.shape
    for i in range(1, height, resolution + 2):
        for j in range(1, width, resolution + 2):
            emoji = png[j:j + resolution, i:i + resolution]
            if np.max(emoji[:, :, -1] != 0):
                emojis.append(emoji)
    return np.array(emojis)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('parse_datasets')

    base_filepath = 'img/'
    companies = ['google', 'apple', 'twitter', 'facebook', 'messenger']
    resolutions = [16, 20, 32, 64]

    categories_data = json.load(open('categories.json', 'r'))
    categories_names = list(categories_data.keys())
    emoji_data = json.load(open('emoji.json', 'r'))

    np.save('companies_names', np.array(companies))
    np.save('categories_names.npy', np.array(categories_names))

    logger.info('generating datasets...')
    for resolution in resolutions:
        dataset = []
        classes = []

        companies_pngs = []
        for company in companies:
            sheet_filepath = base_filepath + ('_'.join(['sheet', company, str(resolution)])) + '.png'
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
        dataset = dataset * 2 - 1

        classes = np.array(classes)

        np.save('emojis_' + str(resolution) + '.npy', dataset)
        np.save('emojis_classes' + '.npy', classes)

    logger.info('done')


if __name__ == "__main__":
    main()
