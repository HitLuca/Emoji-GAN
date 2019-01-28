import os
import shutil

import pytest
from dataset import setup_dataset

companies = ['apple', 'facebook', 'google', 'messenger', 'twitter']
resolutions = [16, 20, 32, 64]
images_dir = 'img/'


@pytest.fixture
def no_images():
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)


@pytest.fixture
def no_datasets():
    dir_name = '../'
    dataset_files = os.listdir('..')

    for item in dataset_files:
        if item.endswith('.npy') or item.endswith('.json'):
            os.remove(os.path.join(dir_name, item))


@pytest.fixture(scope="session", autouse=True)
def change_dir():
    os.chdir('dataset/')


def test_download_images(no_images):
    setup_dataset.main()

    images_present = True

    for company in companies:
        for resolution in resolutions:
            filename = '_'.join(['sheet', company, str(resolution)]) + '.png'
            if not os.path.exists(images_dir + filename):
                images_present = False

    assert images_present
