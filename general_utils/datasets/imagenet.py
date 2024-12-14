import os
from os import path
from io import BytesIO
from typing import Any, Optional, Tuple, List
from PIL import Image
from multiprocessing import Pool
from functools import partial

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

IMAGENET_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

IMAGENET_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def imagenet(root: str,
             train: bool,
             transform: Optional[torch.nn.Module] = None,
             target_transform: Optional[torch.nn.Module] = None,
             download: Optional[bool] = False,
             use_default_transform: Optional[bool] = True
) -> VisionDataset:
    """"""

    if train:
        root = os.path.join(root, "training_data")
    else:
        root = os.path.join(root, "validation_data")

    # Check if root directory exists
    if not path.isdir(root) and download:
        print("ImageNet does not support download=True.")
        print(f"Please download Imagenet from https://www.image-net.org/download.php to {root}")
        raise Exception("Imagenet does not support download=True.")

    if transform is None and use_default_transform:
        transform = IMAGENET_TRAIN_TRANSFORM if train else IMAGENET_VAL_TRANSFORM

    return ImageFolder(root, transform, target_transform)


class ImageNetMem(VisionDataset):

    def _load(self, img_folder: str, root_folder: str, all_folders: List[str]) -> None:
        img_files = os.listdir(os.path.join(root_folder, img_folder))
        label = all_folders.index(img_folder)
        contents = []
        for img_file in img_files:
            with open(os.path.join(root_folder, img_folder, img_file), "rb") as img_content:
                img = BytesIO(img_content.read())
                # img = np.array(Image.open(os.path.join(root_folder, img_folder, img_file)))
                # img = np.float16(img)
                # self.dataset.append((img, label))
                # container.append((img, label))
                contents.append((img, label))
        return contents

    def __init__(self,
                 root: str,
                 transforms: Optional[torch.nn.Module] = None,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[torch.nn.Module] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.dataset = []

        class_folders = os.listdir(root)
        # class_folders = class_folders[:10] # TODO: remove, only for testing

        print(f"Loading Imagenet dataset to memory from {root}")
        with Pool(8) as pool:
            content = pool.map(partial(self._load, root_folder=root, all_folders=class_folders), class_folders)
            for c in content:
                self.dataset.extend(c)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.dataset[index]
        image.seek(0)
        image = Image.open(image).convert("RGB")

        # print("here we go 2")
        if self.transform is not None:
            image = self.transform(image)

        # print("here we gop")

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.dataset)


def imagenet_mem(root: str,
                 train: bool,
                 transform: Optional[torch.nn.Module] = None,
                 target_transform: Optional[bool] = None,
                 download: Optional[bool] = False,
                 use_default_transform: Optional[bool] = False
) -> VisionDataset:
    """"""

    if train:
        root = os.path.join(root, "training_data")
    else:
        root = os.path.join(root, "validation_data")

    # Check if root directory exists
    if not path.isdir(root) and download:
        print("ImageNet does not support download=True.")
        print(f"Please download Imagenet from https://www.image-net.org/download.php to {root}")
        raise Exception("Imagenet does not support download=True.")

    if transform is None and use_default_transform:
        transform = IMAGENET_TRAIN_TRANSFORM if train else IMAGENET_VAL_TRANSFORM

    return ImageNetMem(root, None, transform, target_transform)
