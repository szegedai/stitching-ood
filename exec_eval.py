import sys
import argparse

import torch
from torch.utils.data import DataLoader

from general_utils.models import get_model
from general_utils.datasets import DATASET_META
from general_utils.eval import accuracy_loader
from general_utils.general import parse_float_type


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Neural network evaluation")

    parser.add_argument("model_path", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-b", "--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    # parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")

    return parser.parse_args(args)


def run(conf):
    n_classes = DATASET_META[conf.dataset]["n_classes"]
    n_channels = DATASET_META[conf.dataset]["n_channels_in"]
    dataset_func = DATASET_META[conf.dataset]["dataset_func"]

    model = get_model(conf.model_type, n_classes, n_channels)
    model.load_state_dict(torch.load(conf.model_path, map_location="cpu"))
    if conf.gpu >= 0: model.to(f"cuda:{conf.gpu}")

    val_dataset = dataset_func(root=conf.dataset_root, train=False, download=True, use_default_transform=True)

    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=16, prefetch_factor=8)

    clean_acc = accuracy_loader(model, val_loader)

    print(f"Clean accuracy: {clean_acc * 100:.2f}%")


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)

