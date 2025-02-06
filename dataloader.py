import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A

from pathlib import Path


class BUSI_Dataset(Dataset):
    def __init__(self, data_dir, in_shape=(512, 512), transform=True, mask=True, normalize_args='mean_std.csv'):
        self.data_dir = data_dir
        self.mask = mask
        self.images = sorted(Path('%s/images/' % data_dir).glob('*.png'))
        self.labels = sorted(Path('%s/labels/' % data_dir).glob('*.pth'))

        normalize_args = np.loadtxt(normalize_args, delimiter=",")  # mean, std

        if transform:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=7),
                    A.GaussNoise(std_range=(0, 0.3), mean_range=(0, 0), per_channel=False),
                ], p=0.5),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.Normalize(mean=(normalize_args[0]/255).tolist(), std=(normalize_args[1]/255).tolist()),
                A.PadIfNeeded(min_height=512, min_width=512),
                A.RandomCrop(height=512, width=512),
                A.Resize(height=in_shape[0], width=in_shape[1]),
                A.ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=normalize_args[0].tolist(), std=normalize_args[1].tolist()),
                A.PadIfNeeded(min_height=512, min_width=512),
                A.RandomCrop(height=512, width=512),
                A.Resize(height=in_shape[0], width=in_shape[1]),
                A.ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]

        image = cv2.imread(str(image_path))
        label = torch.load(str(label_path), weights_only=False)

        transformed = self.transform(image=image, mask=label['mask']) if self.mask else self.transform(image=image)

        return transformed['image'], {'label': label['label'], 'mask': transformed['mask']} if self.mask else {'label': label['label']}
