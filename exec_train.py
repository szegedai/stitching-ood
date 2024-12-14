import sys
import os
import json
import argparse

import torch

from general_utils.bin import train
from general_utils.general import set_random_seed, parse_float_type


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Neural network training")

    # General
    parser.add_argument("model_type", type=str)
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-o", "--optimizer", type=str, default="adam", dest="optimizer")
    parser.add_argument("-b", "--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, dest="n_epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, dest="weight_decay")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")
    parser.add_argument("-sched", "--lr-scheduler", type=str, default="", dest="scheduler")
    parser.add_argument("-rlre", "--reduce-lr-epochs", nargs="*", type=int, default=[], dest="reduce_lr_epochs")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    parser.add_argument("--grad-clip", type=float, default=None, dest="grad_clip")
    parser.add_argument("-workers", "--loader-workers", type=int, default=1, dest="n_workers")
    parser.add_argument("-prefetch", "--prefetch-factor", type=int, default=1, dest="prefetch_factor")

    # Model saving
    parser.add_argument("-dir", "--save-dir", type=str, required=True, dest="save_dir")
    parser.add_argument("-sf", "--save-frequency", type=int, default=1, dest="save_frequency")

    return parser.parse_args(args)


def run(conf):
    set_random_seed(conf.seed)

    train(model=conf.model_type,
          dataset_name=conf.dataset,
          n_epochs=conf.n_epochs,
          batch_size=conf.batch_size,
          optimizer=conf.optimizer,
          lr=conf.learning_rate,
          wd=conf.weight_decay,
          loss_fn=torch.nn.CrossEntropyLoss(),
          schedulers=[conf.scheduler],
          reduce_lr_epochs=conf.reduce_lr_epochs,
          gradient_clip_threshold=conf.grad_clip,
          device=f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu",
          dataset_root=conf.dataset_root,
          n_workers=conf.n_workers,
          prefetch_factor=conf.prefetch_factor,
          save_folder=conf.save_dir,
          save_frequency=conf.save_frequency)

    # Write parameter configuration
    config_file = os.path.join(conf.save_dir, "config.txt")
    with open(config_file, "w") as f:
        json.dump(conf.__dict__, f, indent=4)


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)

