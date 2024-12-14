from os import path
from typing import Callable, Optional, List, Any, Tuple

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset


OE_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
])


class OutlierExposure(VisionDataset):
    def __init__(self,
                 root: str,
                 train: Optional[bool] = True,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[torch.nn.Module] = None,
                 download: Optional[bool] = False
    ) -> None:
        super().__init__(root, None, transform, target_transform)
        # Check if root directory exists
        root_dir = path.join(root, "oe")
        source_file = path.join(root_dir, "300K_random_images.npy")
        if (not path.isdir(root_dir) or not path.isfile(source_file)) and download:
            print("Outlier Exposure currently does not support download=True.")
            print(f"Please download 300K random images from https://github.com/hendrycks/outlier-exposure to {root_dir}")
            raise Exception("Outlier Exposure currently does not support download=True.")

        if target_transform is not None:
            print("Warning: 300K random images (outlier exposure) is an "
                  "unlabeled dataset and therefore does not support "
                  "target_transform.")

        if not train:
            print("Warning: 300K random images (outlier images) does not have "
                  "a dedicated validation/test set. Even with train=False, the "
                  "train set will be used.")

        self.images = torch.from_numpy(np.load(source_file))
        self.images = self.images.permute(0, 3, 1, 2) # NHWC -> NCHW

    def __getitem__(self, index: int) -> Any:
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, 1 # For compatibility with other supervised loaders

    def __len__(self) -> int:
        return len(self.images)


def outlier_exposure(root: str,
                     train: bool,
                     transform: Optional[torch.nn.Module] = None,
                     target_transform: Optional[torch.nn.Module] = None,
                     download: Optional[bool] = True,
                     use_default_transform: Optional[bool] = True
) -> OutlierExposure:
    """"""

    if transform is None and use_default_transform:
        transform = OE_TRAIN_TRANSFORM if train else transforms.ToTensor()

    return OutlierExposure(root, train, transform, target_transform, download)
