import argparse
import sys
import datetime
import os
import json
from functools import partial
from typing import Optional, List, Dict, Callable, Tuple
from pathlib import Path

import torch
import numpy as np
import sklearn.metrics as sk
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from general_utils.bin import train
from general_utils.datasets import DATASET_META
from general_utils.eval import accuracy_loader
from general_utils.general import set_random_seed, parse_float_type
from general_utils.models import get_model

from nn_stitching.models import StitchedModelWrapper
from nn_stitching.stitchers import Stitcher
from nn_stitching.stitching_layers import get_stitching_layer, ConvToConvStitchingLayer, TransToTransStitchingLayer, ResizedConvToConvStitchingLayer, TransToConvStitchingLayer
from nn_stitching.utils import get_internal_activation
from nn_stitching.utils.ood import get_ood_scores, eval_representation_ood_detection, get_repr_ood_detector
from nn_stitching.utils.stitched_layers import get_stitched_layers


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Stitched representation OOD evaluation")

    parser.add_argument("source_dir", type=str)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-b", "--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    # Baseline dataset for getting ID scores
    parser.add_argument("-idd", "--id-dataset", type=str, default=None, dest="id_dataset")
    parser.add_argument("-iddata", "--id-dataset-root", type=str, default="./data/pytorch", dest="id_dataset_root")

    # OOD detector folder
    parser.add_argument("-detectors", "--ood-detector-dir", type=str, default="models/repr_cls_ood_detectors", dest="ood_detector_dir")

    # Sanity check
    parser.add_argument("-scd", "--sanity-check-dataset", type=str, default=None, dest="sanity_check_dataset")
    parser.add_argument("-scdata", "--sanity-check-dataset-root", type=str, default="./data/pytorch", dest="sanity_check_dataset_root")

    # Save energy scores
    parser.add_argument("-ed", "--energy-dir", type=str, default="results/energies", dest="energy_dir")


    conf = parser.parse_args(args)
    if conf.id_dataset is None:
        conf.id_dataset = conf.dataset
        conf.id_dataset_root = conf.dataset_root

    return conf


def _get_layers_from_summary(summary_file_path: str) -> Tuple[str, str]:
    front_layer = ""
    end_layer = ""

    with open(summary_file_path, "r") as f:
        contents = f.readlines()
        for line in contents:
            if line.startswith("Front layer"):
                front_layer = line.split(" ")[2]
            elif line.startswith("End layer"):
                end_layer = line.split(" ")[2]

    return front_layer, end_layer


def _get_models_from_summary(summary_file_path: str) -> Tuple[str, str]:
    front_model = ""
    end_model = ""

    with open(summary_file_path, "r") as f:
        contents = f.readlines()
        for line in contents:
            if line.startswith("Front model"):
                front_model = line.split(" ")[2]
            elif line.startswith("End model"):
                end_model = line.split(" ")[2]

    return front_model, end_model


def _get_stitching_config(dir: str) -> Dict[str, str]:
    config_file = os.path.join(dir, "config.txt")
    config_obj = None
    with open(config_file, "r") as f:
        config_obj = json.load(f)
    return config_obj


def run(conf):

    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"

    result_subdirs = [dir for dir in sorted(Path(conf.source_dir).iterdir(),
                                            key=os.path.getmtime)
                                  if os.path.isdir(dir)]
    sample_stitching_config = _get_stitching_config(result_subdirs[0])
    front_model_path = sample_stitching_config["front_model_path"]
    end_model_path = sample_stitching_config["end_model_path"]
    front_model_type = sample_stitching_config["front_model_type"]
    end_model_type = sample_stitching_config["end_model_type"]
    front_layers = get_stitched_layers(front_model_type)
    end_layers = get_stitched_layers(end_model_type)

    acc_mtx = [[0]*len(end_layers) for _ in range(len(front_layers))]
    auroc_mtx = [[0]*len(end_layers) for _ in range(len(front_layers))]
    aupr_mtx = [[0]*len(end_layers) for _ in range(len(front_layers))]
    fpr_mtx = [[0]*len(end_layers) for _ in range(len(front_layers))]

    # Load ID dataset
    id_dataset_func = DATASET_META[conf.id_dataset]["dataset_func"]
    id_dataset = id_dataset_func(conf.id_dataset_root, train=False)
    id_data_loader = DataLoader(id_dataset, batch_size=conf.batch_size, shuffle=False)

    # Load eval dataset
    dataset_func = DATASET_META[conf.dataset]["dataset_func"]
    dataset = dataset_func(conf.dataset_root, train=False)
    data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=False)
    ood_num_examples = min(len(dataset), len(id_dataset))

    # Load end model for baseline ID scores
    n_classes = DATASET_META[conf.dataset]["n_classes"]
    n_channels = DATASET_META[conf.dataset]["n_channels_in"]
    input_shape = DATASET_META[conf.dataset]["shape"]
    end_model = get_model(end_model_type, n_classes, n_channels)
    end_model.load_state_dict(torch.load(end_model_path, map_location="cpu"))
    end_model.eval()
    end_model.to(device)

    # Sanity checking metrics
    sc_auroc = [""] * len(end_layers)
    sc_aupr = [""] * len(end_layers)
    sc_fpr = [""] * len(end_layers)

    # Evaluating every layer separately
    for end_layer in end_layers:
        print(f"Evaluating layer {end_layer}")
        detector = get_repr_ood_detector(conf.ood_detector_dir,
                                         end_model_path,
                                         end_layer)
        detector.eval()
        detector.to(device)
        generator = partial(get_internal_activation, model=end_model, layer_name=end_layer)
        id_scores, _, _ = get_ood_scores(detector,
                                         id_data_loader,
                                         conf.batch_size,
                                         ood_num_examples,
                                         generator,
                                         in_dist=True,
                                         device=device)

        energy_dir = f"{conf.energy_dir}/{end_layer}"
        os.makedirs(energy_dir, exist_ok=True)

        print(np.mean(id_scores))

        # Sanity check on OOD dataset
        if conf.sanity_check_dataset is not None:
            metrics = eval_representation_ood_detection(end_model,
                                                        end_layer,
                                                        detector,
                                                        id_scores=id_scores,
                                                        ood_dataset=conf.sanity_check_dataset,
                                                        ood_dataset_root=conf.sanity_check_dataset_root,
                                                        batch_size=conf.batch_size,
                                                        device=device,
                                                        generator=generator,
                                                        num_to_avg=1,
                                                        save_to=f"{energy_dir}/{conf.sanity_check_dataset}.npy")
            auroc, aupr, fpr = metrics
            layer_to_idx = end_layers.index(end_layer)
            sc_auroc[layer_to_idx] = str(auroc)
            sc_aupr[layer_to_idx] = str(aupr)
            sc_fpr[layer_to_idx] = str(fpr)

            print(f"Sanity checking {end_layer}'s OOD detector:")
            print(f"AUROC: {auroc}")
            print(f"AUPR: {aupr}")
            print(f"FPR95: {fpr}")
            print()

        # Evaluate stitching results
        for dir in result_subdirs:
            summary_file = os.path.join(dir, "summary.txt")
            model_file = os.path.join(dir, "final_model.pt")

            front_l, end_l = _get_layers_from_summary(summary_file)
            fm, em = _get_models_from_summary(summary_file)
            if end_l != end_layer: continue

            # Setup stitched model
            front_m = get_model(front_model_type, n_classes, n_channels)
            end_m = get_model(end_model_type, n_classes, n_channels)
            front_m.load_state_dict(torch.load(fm, map_location="cpu"))
            end_m.load_state_dict(torch.load(em, map_location="cpu"))
            front_wrapper = StitchedModelWrapper(front_m,
                                                 input_shape=input_shape,
                                                 stitch_from=front_l)
            end_wrapper = StitchedModelWrapper(end_m,
                                               input_shape=input_shape,
                                               stitch_to=end_l)

            stitching_config = _get_stitching_config(dir)
            stitching_layer = get_stitching_layer(**stitching_config)

            stitcher = Stitcher(front_wrapper, end_wrapper, stitching_layer)
            stitcher.stitching_layer.load_state_dict(torch.load(model_file, map_location="cpu"))
            stitcher.eval()
            stitcher.to(device)

            # Evaluate
            acc = accuracy_loader(stitcher, data_loader)

            metrics = eval_representation_ood_detection(stitcher,
                                                        "stitching_layer",
                                                        detector,
                                                        id_scores=id_scores,
                                                        ood_loader=data_loader,
                                                        batch_size=conf.batch_size,
                                                        device=device,
                                                        num_to_avg=1,
                                                        save_to=f"{energy_dir}/{front_l}.npy")
            auroc, aupr, fpr = metrics

            print(f"{front_l}")
            print(f"Accuracy: {acc}")
            print(f"AUROC: {auroc}")
            print(f"AUPR: {aupr}")
            print(f"FPR95: {fpr}")
            print()

            layer_from_idx = front_layers.index(front_l)
            layer_to_idx = end_layers.index(end_l)

            acc_mtx[layer_from_idx][layer_to_idx] = acc
            auroc_mtx[layer_from_idx][layer_to_idx] = auroc
            aupr_mtx[layer_from_idx][layer_to_idx] = aupr
            fpr_mtx[layer_from_idx][layer_to_idx] = fpr

    # Print results
    print("STITCHING ACCURACY")
    print(acc_mtx)
    # for i in range(len(front_layers)):
    #     print(";".join(acc_mtx[i]))

    print("STITCHING OOD Detection AUROC")
    print(auroc_mtx)
    # for i in range(len(front_layers)):
    #     print(";".join(auroc_mtx[i]))

    print("STITCHING OOD Detection AUPR")
    print(aupr_mtx)
    # for i in range(len(front_layers)):
    #     print(";".join(aupr_mtx[i]))

    print("STITCHING OOD Detection FPR95")
    print(fpr_mtx)
    # for i in range(len(front_layers)):
    #     print(";".join(fpr_mtx[i]))

    if conf.sanity_check_dataset is not None:
        print(f"OOD detector sanity checks: {conf.sanity_check_dataset}")
        print(f"AUROC: {';'.join(sc_auroc)}")
        print(f"AUPR: {';'.join(sc_aupr)}")
        print(f"FPR95: {';'.join(sc_fpr)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
