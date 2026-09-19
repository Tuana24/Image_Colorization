import os
import random
import numpy as np

from PIL import Image
from skimage.color import rgb2lab

import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):

    def __init__(
        self,
        image_folder,
        image_size=96,
        augment=False
    ):

        self.image_folder = image_folder
        self.image_size = image_size
        self.augment = augment

        self.image_files = []

        for filename in os.listdir(image_folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                self.image_files.append(
                    os.path.join(
                        image_folder,
                        filename
                    )
                )

        self.image_files.sort()


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, index):

        image_path = self.image_files[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        # Resize image
        image = image.resize(
            (self.image_size, self.image_size)
        )
        # Optional augmentation
        if self.augment:

            if random.random() > 0.5:
                image = image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

        # Convert image to numpy
        image_rgb = np.array(
            image
        ).astype(np.float32) / 255.0

        # RGB -> LAB
        image_lab = rgb2lab(
            image_rgb
        ).astype(np.float32)

        # L channel
        L = image_lab[:, :, 0]

        # a and b channels
        ab = image_lab[:, :, 1:]

        # Normalize
        L = L / 100.0

        ab = ab / 128.0

        # Convert L to PyTorch tensor
        L = torch.from_numpy(
            L
        ).unsqueeze(0)

        # Convert ab to PyTorch tensor
        ab = torch.from_numpy(
            ab
        ).permute(2, 0, 1)


        return L, ab


if __name__ == "__main__":

    dataset = ColorizationDataset(
        "data/train_landscape",
        image_size=96
    )

    print(
        "Number of images:",
        len(dataset)
    )

    L, ab = dataset[0]

    print(
        "L shape:",
        L.shape
    )

    print(
        "ab shape:",
        ab.shape
    )

    print(
        "L minimum:",
        L.min().item()
    )

    print(
        "L maximum:",
        L.max().item()
    )

    print(
        "ab minimum:",
        ab.min().item()
    )

    print(
        "ab maximum:",
        ab.max().item()
    )