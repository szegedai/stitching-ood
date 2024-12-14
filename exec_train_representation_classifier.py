import sys
import os
import json
import argparse
import uuid

import torch
import torch.nn.functional as F
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from general_utils.models import get_model
from general_utils.datasets import DATASET_META
from general_utils.general import set_random_seed, get_layer, parse_float_type
from general_utils.train.samplers import MultiEpochDisjointSubsetSampler


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Internal representation (activation) classifier & OOD detector trainer")

    # Source model
    parser.add_argument("source_model_path", type=str)
    parser.add_argument("source_model_type", type=str)
    parser.add_argument("source_layer_name", type=str)

    # Representation classifier
    parser.add_argument("-m", "--model-type", type=str, dest="model_type")

    # Base training parameters
    parser.add_argument("-d", "--dataset", type=str, dest="dataset")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    parser.add_argument("-e", "--epochs", type=int, default=100, dest="n_epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5, dest="weight_decay")
    parser.add_argument("-rlre", "--reduce-lr-epochs", nargs="*", type=int, default=[], dest="reduce_lr_epochs")
    parser.add_argument("-dir", "--save-dir", type=str, required=True, dest="save_dir")
    parser.add_argument("-sf", "--save-frequency", type=int, default=1, dest="save_frequency")

    # OOD detection finetuning parameters
    parser.add_argument("-ftidd", "--finetune-id-dataset", type=str, dest="finetune_id_dataset")
    parser.add_argument("-ftiddata", "--finetune-id-dataset-root", type=str, default="./data/pytorch", dest="finetune_id_dataset_root")
    parser.add_argument("-ftoodd", "--finetune-ood-dataset", type=str, dest="finetune_ood_dataset")
    parser.add_argument("-ftooddata", "--finetune-ood-dataset-root", type=str, default="./data/pytorch", dest="finetune_ood_dataset_root")
    parser.add_argument("-fe", "--finetune-epochs", type=int, default=20, dest="n_finetune_epochs")
    parser.add_argument("-m-in", "--mean-in-score", type=float, default=-25, dest="m_in")
    parser.add_argument("-m-out", "--mean-out-score", type=float, default=-7, dest="m_out")
    parser.add_argument("-score", "--ood-score", type=str, default="energy", dest="score")
    parser.add_argument("-fdir", "--final-save-dir", type=str, default="./models/repr_cls_ood_detectors", dest="final_save_dir")

    return parser.parse_args(args)


def run(conf):

    set_random_seed(0)

    # setup dataset and dataloaders
    n_classes = DATASET_META[conf.dataset]["n_classes"]
    n_channels = DATASET_META[conf.dataset]["n_channels_in"]
    input_shape = DATASET_META[conf.dataset]["shape"]
    dataset_func = DATASET_META[conf.dataset]["dataset_func"]

    train_dataset = dataset_func(conf.dataset_root, train=True)
    val_dataset = dataset_func(conf.dataset_root, train=False)
    train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False)

    # setup source model
    source_model = get_model(conf.source_model_type, n_classes, n_channels)
    source_model.load_state_dict(torch.load(conf.source_model_path, map_location="cpu"))
    source_model.eval()

    # get input shape
    source_layer = get_layer(source_model, conf.source_layer_name)
    repr_shape = None
    def repr_shape_store_hook(m, i, o):
        nonlocal repr_shape
        repr_shape = o.size()[1:]
    shape_hook = source_layer.register_forward_hook(repr_shape_store_hook)
    source_model(torch.rand(2, *input_shape))
    shape_hook.remove()

    # setup activation storage hook
    activation = None
    def act_store_hook(m, i, o):
        nonlocal activation
        activation = o.detach()
    act_hook = source_layer.register_forward_hook(act_store_hook)

    # setup model
    model = get_model(conf.model_type,
                      n_classes,
                      repr_shape[0],
                      orig_input_shape=input_shape,
                      repr_input_shape=repr_shape)
    model.train()

    # train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, conf.reduce_lr_epochs, 0.1)

    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"
    source_model.to(device)
    model.to(device)

    training_logs = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": []
    }
    os.makedirs(conf.save_dir, exist_ok=True)

    for epoch in range(1, conf.n_epochs + 1):

        # Train step
        model.train()
        n_samples = 0
        sum_acc = 0
        sum_loss = 0

        for img, label in tqdm(train_loader):
            optimizer.zero_grad()

            img, label = img.to(device), label.to(device)

            source_model(img)
            pred = model(activation)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            sum_loss += loss.item() * len(label)
            sum_acc += torch.argmax(pred, 1).eq(label).sum().item()
            n_samples += len(label)

        scheduler.step()

        training_logs["train_acc"].append(sum_acc / n_samples)
        training_logs["train_loss"].append(sum_loss / n_samples)
        print(f"Epoch #{epoch} train loss: {sum_loss / n_samples}, accuracy: {sum_acc / n_samples}")

        # Val step
        model.eval()
        n_samples = 0
        sum_acc = 0
        sum_loss = 0

        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            source_model(img)
            pred = model(activation)
            loss = loss_fn(pred, label)

            sum_loss += loss.item() * len(label)
            sum_acc += torch.argmax(pred, 1).eq(label).sum().item()
            n_samples += len(label)

        training_logs["val_acc"].append(sum_acc / n_samples)
        training_logs["val_loss"].append(sum_loss / n_samples)
        print(f"Epoch #{epoch} val loss: {sum_loss / n_samples}, accuracy: {sum_acc / n_samples}")

        # model saving, log updating, etc.
        log_file = os.path.join(conf.save_dir, "training_logs.csv")
        pd.DataFrame(training_logs).to_csv(log_file, index=False)

        if epoch % conf.save_frequency == 0:
            checkpoint_file = os.path.join(conf.save_dir, f"model_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    # Finetuning datasets
    id_dataset_func = DATASET_META[conf.finetune_id_dataset]["dataset_func"]
    id_dataset = id_dataset_func(conf.finetune_id_dataset_root, train=True)

    ood_dataset_func = DATASET_META[conf.finetune_ood_dataset]["dataset_func"]
    ood_dataset = ood_dataset_func(conf.finetune_ood_dataset_root, train=True)

    n_samples_per_epoch = min(len(id_dataset), len(ood_dataset))
    id_sampler = MultiEpochDisjointSubsetSampler(id_dataset, n_samples=n_samples_per_epoch)
    ood_sampler = MultiEpochDisjointSubsetSampler(ood_dataset, n_samples=n_samples_per_epoch)

    id_loader = DataLoader(id_dataset, batch_size=conf.batch_size // 2, sampler=id_sampler)
    ood_loader = DataLoader(ood_dataset, batch_size=conf.batch_size // 2, sampler=ood_sampler)

    # Finetuning for OOD detection
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [conf.n_finetune_epochs // 2], 0.1)

    ft_logs = {
        "ft_loss": [],
        "ce_loss": [],
        "train_acc": [],
        "val_ce_loss": [],
        "val_acc": []
    }

    for epoch in range(1, conf.n_finetune_epochs + 1):

        model.train()
        n_samples = 0
        n_id_samples = 0
        sum_ft_loss = 0
        sum_ce_loss = 0
        sum_acc = 0

        for in_set, out_set in tqdm(zip(id_loader, ood_loader)):
            in_x, in_y = in_set
            in_x, in_y = in_x.to(device), in_y.to(device)

            in_x, in_y = in_x.cpu().detach(), in_y.cpu().detach()
            data = torch.cat((in_x, out_set[0]), 0)
            target = in_y

            data, target = data.to(device), target.to(device)

            # Activation storage hook is still attached
            source_model(data)

            # Get outputs
            x = model(activation)

            # backward
            optimizer.zero_grad()

            loss = F.cross_entropy(x[:len(in_set[0])], target)
            # cross-entropy from softmax distribution to uniform distribution
            if conf.score == 'energy':
                Ec_out = -torch.logsumexp(x[len(in_set[0]):], dim=1)
                Ec_in = -torch.logsumexp(x[:len(in_set[0])], dim=1)
                loss += 0.1*(torch.pow(F.relu(Ec_in-conf.m_in), 2).mean() + torch.pow(F.relu(conf.m_out-Ec_out), 2).mean())
            elif conf.score == 'OE':
                loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            sum_ft_loss += loss.item() * len(data)
            sum_ce_loss += F.cross_entropy(x[:len(in_set[0])], target).item() * len(target)
            sum_acc += torch.argmax(x[:len(in_set[0])], 1).eq(target).sum().item()
            n_samples += len(data)
            n_id_samples += len(target)

        ft_logs["ft_loss"].append(sum_ft_loss / n_samples)
        ft_logs["ce_loss"].append(sum_ce_loss / n_samples)
        ft_logs["train_acc"].append(sum_acc / n_id_samples)
        print(f"FT Epoch #{epoch} FT loss: {sum_ft_loss / n_samples}, CE Loss: {sum_ce_loss / n_samples}, accuracy: {sum_acc / n_id_samples}")

        # Val step
        model.eval()
        n_samples = 0
        sum_acc = 0
        sum_loss = 0

        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            source_model(img)
            pred = model(activation)
            loss = loss_fn(pred, label)

            sum_loss += loss.item() * len(label)
            sum_acc += torch.argmax(pred, 1).eq(label).sum().item()
            n_samples += len(label)

        ft_logs["val_acc"].append(sum_acc / n_samples)
        ft_logs["val_ce_loss"].append(sum_loss / n_samples)
        print(f"FT Epoch #{epoch} val loss: {sum_loss / n_samples}, accuracy: {sum_acc / n_samples}")

        # model saving, log updating, etc.
        log_file = os.path.join(conf.save_dir, "ft_logs.csv")
        pd.DataFrame(ft_logs).to_csv(log_file, index=False)

        checkpoint_file = os.path.join(conf.save_dir, f"ft_model_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    # Write parameter configuration
    config_file = os.path.join(conf.save_dir, "config.txt")
    with open(config_file, "w") as f:
        json.dump(conf.__dict__, f, indent=4)

    # Final model and metadata saving
    model_id = uuid.uuid4().hex
    final_model_file = os.path.join(conf.final_save_dir, f"{model_id}.pt")
    descriptor_file = os.path.join(conf.final_save_dir, "meta.json")
    metadata_key = f"{model_id}.pt"
    metadata_obj = {
            "model_type": conf.model_type,
            "source_model_type": conf.source_model_type,
            "source_model_path": conf.source_model_path,
            "source_layer": conf.source_layer_name,
            "id_dataset": conf.dataset,
            "orig_input_shape": input_shape,
            "required_input_shape": repr_shape,
            "training_loc": conf.save_dir
        }

    os.makedirs(conf.final_save_dir, exist_ok=True)
    torch.save(model.state_dict(), final_model_file, _use_new_zipfile_serialization=False)
    if os.path.isfile(descriptor_file):
        metadata = None
        with open(descriptor_file, "r") as f:
            metadata = json.load(f)
        metadata[metadata_key] = metadata_obj
        with open(descriptor_file, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        metadata = {metadata_key: metadata_obj}
        with open(descriptor_file, "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
