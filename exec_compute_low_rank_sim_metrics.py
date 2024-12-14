import sys
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from general_utils.datasets import DATASET_META
from general_utils.general import set_random_seed
from general_utils.models import get_model

from nn_stitching.utils.representational_similarity import compute_metrics


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Low rank CKA computation")

    parser.add_argument("model1_path", type=str)
    parser.add_argument("model2_path", type=str)
    parser.add_argument("model1_type", type=str)
    parser.add_argument("model2_type", type=str)
    parser.add_argument("model1_layer", type=str)
    parser.add_argument("model2_layer", type=str)
    parser.add_argument("-m1r", "--model1-ranks", type=int, default=None, dest="model1_ranks", nargs="*")
    parser.add_argument("-m2r", "--model2-ranks", type=int, default=None, dest="model2_ranks", nargs="*")

    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    parser.add_argument("-workers", "--loader-workers", type=int, default=1, dest="n_workers")
    parser.add_argument("-prefetch", "--prefetch-factor", type=int, default=1, dest="prefetch_factor")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")

    return parser.parse_args(args)


def run(conf):

    set_random_seed(0)

    # Dataset
    n_classes = DATASET_META[conf.dataset]["n_classes"]
    n_channels = DATASET_META[conf.dataset]["n_channels_in"]
    dataset_func = DATASET_META[conf.dataset]["dataset_func"]
    val_dataset = dataset_func(conf.dataset_root, train=False)
    data_loader = DataLoader(val_dataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=conf.n_workers,
                             prefetch_factor=conf.prefetch_factor)

    # Models
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"

    model1 = get_model(conf.model1_type, n_classes, n_channels)
    model2 = get_model(conf.model2_type, n_classes, n_channels)
    model1.load_state_dict(torch.load(conf.model1_path, map_location="cpu"))
    model2.load_state_dict(torch.load(conf.model2_path, map_location="cpu"))
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    m1_ranks = sorted(conf.model1_ranks, reverse=True) if conf.model1_ranks else [None]
    m2_ranks = sorted(conf.model2_ranks, reverse=True) if conf.model2_ranks else [None]

    for m1_rank in m1_ranks:
        ckas = []
        ccas = []
        procrustes = []

        for m2_rank in m2_ranks:

            metrics = compute_metrics(model1,
                                      model2,
                                      conf.model1_layer,
                                      conf.model2_layer,
                                      data_loader,
                                      1000,
                                      device,
                                      metrics=["cka", "pwcca", "procrustes"],
                                      model1_rank=m1_rank,
                                      model2_rank=m2_rank)

            ckas.append(metrics["cka"])
            ccas.append(metrics["pwcca"])
            procrustes.append(metrics["procrustes"])


        print(f"{conf.model1_path} -- {conf.model2_path}")
        print(f"{conf.model1_layer} -- {conf.model2_layer}")
        print("CKA:")
        print(f"Model1 rank {m1_rank}:")
        for m2_rank, cka in zip(m2_ranks, ckas):
            print(f"\tModel2 rank {m2_rank}: {cka}")
        print()

        for cka in ckas:
            print(cka)

        print("\nCCA:")
        # print("---")
        # print("---")
        # print("---")

        print(f"Model1 rank {m1_rank}:")
        for m2_rank, cca in zip(m2_ranks, ccas):
            print(f"\tModel2 rank {m2_rank}: {cca}")
        print()

        for cca in ccas:
            print(cca)

        print("\nOrth. Procr:")
        # print("---")
        # print("---")
        # print("---")

        print(f"Model1 rank {m1_rank}:")
        for m2_rank, procr in zip(m2_ranks, procrustes):
            print(f"\tModel2 rank {m2_rank}: {procr}")
        print()

        for procr in procrustes:
            print(procr)

        print("\n\n")


    # print(f"{conf.model1_layer} (rank {conf.model1_rank}) -- {conf.model2_layer} (rank {conf.model2_rank})")
    # print(f"CKA: {cka}")
    # print("\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
