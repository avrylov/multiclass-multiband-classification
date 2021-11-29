from typing import List, Tuple
import random

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tf


class MyDataset(Dataset):
    def __init__(self, df, is_train=False,
                 s2_input_size=(13, 32, 32)):

        self.is_train = is_train  # flag for augmentation
        self.df = df.copy()  # df with path to files and features
        self.s2_input_size = s2_input_size

    def __len__(self):
        return len(self.df)

    def __data_aug(self, bands: List, output_size: Tuple) -> List:

        angle = random.randint(1, 90)
        translate = [random.randint(0, 2), random.randint(0, 2)]
        scale = round(random.uniform(0.95, 0.99), 2)
        shear = [random.randint(0, 2), random.randint(0, 2)]

        augmented_bands = []
        for band in bands:

            band = torch.tensor(band)
            i, j, h, w = transforms.RandomCrop.get_params(
                band, output_size=output_size)
            band = tf.crop(band, i, j, h, w)

            band = band.unsqueeze(0)
            band = tf.affine(band,
                             angle=angle,
                             translate=translate,
                             scale=scale,
                             shear=shear)
            band = band.squeeze(0)
            augmented_bands.append(band)
        return augmented_bands

    def __get_input(self, row):
        data_bands_13 = np.load(row.s2_file_names)
        amax = np.amax(data_bands_13)
        data_bands_13 = data_bands_13 / amax
        bands_13 = []
        for i in range(self.s2_input_size[0]):
            single_band = data_bands_13[i, :, :]
            single_band = single_band / single_band.max()
            bands_13.append(np.array(single_band))

        if self.is_train:
            bands_13 = self.__data_aug(
                bands=bands_13,
                output_size=(
                    self.s2_input_size[1],
                    self.s2_input_size[2]
                )
            )

        bands_13 = np.dstack(bands_13)
        bands_13 = np.swapaxes(bands_13, 0, 2)
        return bands_13

    def __get_output(self, row):
        return np.argmax(np.array(row.values[2:], dtype=np.float64))

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        inputs = self.__get_input(row)
        y = self.__get_output(row)
        return index, inputs, y

