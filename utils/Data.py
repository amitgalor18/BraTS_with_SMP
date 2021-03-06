from torch.utils.data import DataLoader
import cv2
import pickle
import os
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """BraTS Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['complete','core', 'unlabelled']


    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.images_fps = images_dir
        self.masks_fps = masks_dir

        self.augmentation = augmentation
        self.preprocessing = preprocessing


    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. complete)
        mask = (mask >= 25).astype('uint8')
        mask = np.expand_dims(mask, axis=2)
        mask = mask.astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, self.images_fps[i]


    def __len__(self):
        return len(self.images_fps)