import argparse
import sys
import datetime
import os
import json
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader

from general_utils.bin import train
from general_utils.datasets import DATASET_META
from general_utils.eval import accuracy_loader
from general_utils.general import set_random_seed, parse_float_type
from general_utils.models import get_model

from nn_stitching.models import StitchedModelWrapper
from nn_stitching.stitchers import Stitcher
from nn_stitching.stitching_layers import get_stitching_layer
from nn_stitching.init import least_squares_init
from nn_stitching.utils.sparsity import stitcher_l1_loss, sparsity


# TODO: move to utils
def stitched_repr_diff(stitcher: Stitcher, data_loader: DataLoader, device) -> float:
    normalized_repr_diffs = []

    for data, _ in data_loader:
        data = data.to(device)
        _, activations = stitcher(data, return_activations=True)
        stitched_activation = activations["stitched"]
        end_activation = activations["end"]

        diff = stitched_activation - end_activation
        norm_diff = ((torch.norm(diff, "fro") ** 2) /
                     (torch.norm(end_activation, "fro") ** 2)).item()
        normalized_repr_diffs.append(norm_diff)

    return np.mean(np.array(normalized_repr_diffs))



def _parse_args(args):
    parser = argparse.ArgumentParser(description="Model stitching")

    # General
    parser.add_argument("front_model_path", type=str)
    parser.add_argument("end_model_path", type=str)
    parser.add_argument("front_model_type", type=str)
    parser.add_argument("end_model_type", type=str)
    parser.add_argument("front_layer", type=str)
    parser.add_argument("end_layer", type=str)
    parser.add_argument("stitching_type", type=str, choices=["c2c", "t2t", "c2t", "t2c", "rc2c_pre"])
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-o", "--optimizer", type=str, default="adam", dest="optimizer")
    parser.add_argument("-b", "--batch-size", type=int, default=256, dest="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=30, dest="n_epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, dest="weight_decay")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")
    parser.add_argument("-rlre", "--reduce-lr-epochs", nargs="*", type=int, default=[], dest="reduce_lr_epochs")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    parser.add_argument("-workers", "--loader-workers", type=int, default=1, dest="n_workers")
    parser.add_argument("-prefetch", "--prefetch-factor", type=int, default=1, dest="prefetch_factor")

    # Model saving
    parser.add_argument("-dir", "--save-dir", type=str, default="", dest="save_dir")
    # parser.add_argument("-sf", "--save-frequency", type=int, default=1, dest="save_frequency")
    parser.add_argument("--no-save", action="store_true", default=False, dest="no_save")

    # Ablation, misc
    parser.add_argument("--mod-only-stitch", action="store_true", default=False, dest="mod_only_stitch")
    parser.add_argument("-i", "--init", type=str, default="rand", dest="init")
    parser.add_argument("-pinv-samples", "--pinv-init-samples", type=int, default=100, dest="n_pinv_init_samples")
    parser.add_argument("-l1", "--l1-coef", type=float, default=0, dest="l1_coef")
    parser.add_argument("-c2t-cls", "--c2t-cls-token", type=str, default=None, dest="c2t_cls_token", choices=[None, "pool", "learn"])
    parser.add_argument("-t2c-cls", "--t2c-cls-token", type=str, default=None, dest="t2c_cls_token", choices=[None, "drop", "add", "only"])
    parser.add_argument("-upsample", "--upsample-mode", type=str, default="bilinear", dest="upsample_mode", choices=["bilinear", "nearest"])
    parser.add_argument("--front-stitch-rank", type=int, default=None, dest="front_stitch_rank")
    parser.add_argument("--front-pinv-rank", type=int, default=None, dest="front_pinv_rank")

    return parser.parse_args(args)


def run(conf):
    # Seed
    set_random_seed(conf.seed)

    n_classes = DATASET_META[conf.dataset]["n_classes"]
    n_channels = DATASET_META[conf.dataset]["n_channels_in"]
    input_shape = DATASET_META[conf.dataset]["shape"]

    # Donor models
    front_model = get_model(conf.front_model_type, n_classes, n_channels)
    front_model.load_state_dict(torch.load(conf.front_model_path, map_location="cpu"))
    front_wrapper = StitchedModelWrapper(model=front_model,
                                         input_shape=input_shape,
                                         stitch_from=conf.front_layer)

    end_model = get_model(conf.end_model_type, n_classes, n_channels)
    end_model.load_state_dict(torch.load(conf.end_model_path, map_location="cpu"))
    end_wrapper = StitchedModelWrapper(model=end_model,
                                       input_shape=input_shape,
                                       stitch_to=conf.end_layer)


    # Stitching layer
    stitching_layer = get_stitching_layer(conf.stitching_type,
                                          init=conf.init,
                                          c2t_cls_token=conf.c2t_cls_token,
                                          t2c_cls_token=conf.t2c_cls_token,
                                          upsample_mode=conf.upsample_mode)

    # Stitched model
    stitcher = Stitcher(front_model=front_wrapper,
                        end_model=end_wrapper,
                        stitching_layer=stitching_layer,
                        modify_only_stitching_layer=conf.mod_only_stitch)

    # For monitoring matched difference
    pinv_loss = 0

    # Least squares init if necessary
    if conf.init == "pinv":
        _, pinv_loss = least_squares_init(front_model=stitcher,
                           end_model=StitchedModelWrapper(end_model,
                                                          input_shape,
                                                          conf.end_layer),
                           front_layer_name="stitching_layer.transform",
                           end_layer_name=f"model.{conf.end_layer}",
                           dataset_name=conf.dataset,
                           dataset_root=conf.dataset_root,
                           n_samples=conf.n_pinv_init_samples,
                           stitcher=stitcher,
                           front_pre_activation=True,
                           front_rank=conf.front_pinv_rank)

    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    stitcher.to(device)

    # Add front rank to stitcher now so it does not interfere with initialization
    stitcher.front_rank = conf.front_stitch_rank

    # Save folder
    curr_time = "-".join("_".join(str(datetime.datetime.now()).split(" ")).split(":"))
    if conf.save_dir != "":
        save_folder = os.path.join("results", conf.save_dir, curr_time)
    else:
        save_folder = os.path.join("results", curr_time)
    os.makedirs(save_folder)

    # Loss function
    if conf.l1_coef > 0:
        loss_fn = partial(stitcher_l1_loss,
                          stitcher=stitcher,
                          coef=conf.l1_coef)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    final_model = train(model=stitcher,
                        dataset_name=conf.dataset,
                        n_epochs=conf.n_epochs,
                        batch_size=conf.batch_size,
                        optimizer=conf.optimizer,
                        lr=conf.learning_rate,
                        wd=conf.weight_decay,
                        loss_fn=loss_fn,
                        reduce_lr_epochs=conf.reduce_lr_epochs,
                        device=device,
                        dataset_root=conf.dataset_root,
                        save_folder=save_folder,
                        save_frequency=conf.n_epochs + 1, # don't save here
                        n_workers=conf.n_workers,
                        prefetch_factor=conf.prefetch_factor)

    # Eval
    dataset_func = DATASET_META[conf.dataset]["dataset_func"]
    val_dataset = dataset_func(root=conf.dataset_root,
                               train=False,
                               download=True,
                               use_default_transform=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=conf.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=16,
                            prefetch_factor=8)
    clean_acc = accuracy_loader(final_model, val_loader)

    # Frobenius norm difference between stitched and target representations
    # DM (struct.) in the paper
    norm_repr_diff = stitched_repr_diff(stitcher, val_loader, device)

    # Save final model
    if not conf.no_save:
        if conf.mod_only_stitch or conf.n_epochs == 0:
            torch.save(final_model.stitching_layer.state_dict(),
                       os.path.join(save_folder, "final_model.pt"),
                       _use_new_zipfile_serialization=False)
        else:
            torch.save(final_model.state_dict(),
                       os.path.join(save_folder, "final_model.pt"),
                       _use_new_zipfile_serialization=False)

    # Write results
    result_summary_file = os.path.join(save_folder, "summary.txt")
    with open(result_summary_file, "w") as f:
        f.write(f"Front model: {conf.front_model_path} \n")
        f.write(f"End model: {conf.end_model_path} \n")
        f.write(f"Front layer: {conf.front_layer} \n")
        f.write(f"End layer: {conf.end_layer} \n")
        f.write(f"Clean accuracy: {clean_acc:.4f} \n")
        f.write(f"Normalized PINV difference: {pinv_loss:.4f}\n")
        f.write(f"Normalized representation difference: {norm_repr_diff:.4f}\n\n")
        f.write(f"Command: {' '.join(sys.argv)}\n\n")

        weight = stitcher.stitching_layer.transform.weight
        bias = stitcher.stitching_layer.transform.weight
        f.write(f"Sparsity\n")
        f.write(f"0: \t\t{sparsity([weight, bias], 0)}\n")
        f.write(f"1e-5: \t{sparsity([weight, bias], 1e-5)}\n")
        f.write(f"1e-4: \t{sparsity([weight, bias], 1e-4)}\n")
        f.write(f"1e-3: \t{sparsity([weight, bias], 1e-3)}\n")
        f.write(f"1e-2: \t{sparsity([weight, bias], 1e-2)}\n")
        f.write(f"1e-1: \t{sparsity([weight, bias], 1e-1)}\n")

    # Write parameter configuration
    config_file = os.path.join(save_folder, "config.txt")
    with open(config_file, "w") as f:
        json.dump(conf.__dict__, f, indent=4)

    # Print results just for the sake of it
    print(f"Clean accuracy: {clean_acc * 100:.2f}%")


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
