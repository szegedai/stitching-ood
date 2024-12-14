from typing import Optional, Union, List

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from nn_stitching.utils import get_internal_activation
from nn_stitching.utils.low_rank import low_rank_approx

from .cca import pwcca_dist
from .procrustes import procrustes


def resize_for_cca(x: torch.Tensor):
    # Idea broadly from
    # https://github.com/google/svcca/blob/master/tutorials/002_CCA_for_Convolutional_Layers.ipynb
    # and Raghu et al. SVCCA: Singular Vector Canonical Correlation Analysis
    # for Deep Learning Dynamics and Interpretability (NeurIPS 2017)

    if len(x.shape) == 3:
        # Transformers - use first token (also used by Ding et al.)
        x = x[:, 0, :]

    elif len(x.shape) == 4:
        # Conv.nets - avg pool along spatial dimensions
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = torch.flatten(x, 1)

    return x


def resize_for_cca2(x, y):
    if len(x.shape) == len(y.shape) == 3:
        # x = x[:, 0, :]
        # y = y[:, 0, :]
        if x.shape == y.shape:
            b, n, c = x.shape
            x = x.reshape((b*n, c))
            y = y.reshape((b*n, c))
        else:
            if x.shape[1] == y.shape[1]:
                b, n1, c = x.shape
                b, n2, c = y.shape
                x = x.reshape((b*n1, c))
                y = y.reshape((b*n2, c))
            else:
                x = x.mean(dim=1)
                y = y.mean(dim=1)

    elif len(x.shape) == len(y.shape) == 4:
        if x.shape == y.shape:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape((b*h*w, c))
            y = y.permute(0, 2, 3, 1).reshape((b*h*w, c))
        else:
            # As per Google's recommendation
            # TODO: link to SVCCA repo tutorial
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = torch.flatten(x, 1)
            y = nn.AdaptiveAvgPool2d(1)(y)
            y = torch.flatten(y, 1)

    return x, y


def compute_metrics(model1: nn.Module,
                    model2: nn.Module,
                    layer1: str,
                    layer2: str,
                    data_loader: DataLoader,
                    n_samples: int,
                    device: Union[torch.device, str],
                    metrics: Union[str, List[str]] = "cka",
                    cross_task: bool = False,
                    model1_rank: Optional[int] = None,
                    model2_rank: Optional[int] = None,
) -> float:

    # convert single metric to a list
    if type(metrics) == str:
        metrics = [metrics]

    act1 = []
    act2 = []

    metric_values = {}

    # get internal representations
    for data in data_loader:
        if cross_task:
                x1, x2, y = data
        else:
            x1, y = data
            x2 = x1.clone()

        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        a1 = get_internal_activation(x1, y, model1, layer1).detach().cpu()
        a2 = get_internal_activation(x2, y, model2, layer2).detach().cpu()

        act1.extend(a1)
        act2.extend(a2)

        if len(act1) >= n_samples:
            act1 = act1[:n_samples]
            act2 = act2[:n_samples]
            break

    act1 = torch.stack(act1)
    act2 = torch.stack(act2)

    if model1_rank: act1 = low_rank_approx(act1, model1_rank)
    if model2_rank: act2 = low_rank_approx(act2, model2_rank)

    # Resize and transpose for CCA & OPD
    act1_cca_opd, act2_cca_opd = resize_for_cca2(act1, act2)
    act1_cca_opd = act1_cca_opd.numpy().T
    act2_cca_opd = act2_cca_opd.numpy().T

    act1 = torch.flatten(act1, 1).numpy()
    act2 = torch.flatten(act2, 1).numpy()


    for metric in metrics:
        if metric == "cka":
            from .cka import cka
            val = cka(act1, act2)

        elif metric == "pwcca":
            from .cca import cca_decomp
            # normalize activations first
            act1_norm = act1_cca_opd - act1_cca_opd.mean(axis=1, keepdims=True)
            act2_norm = act2_cca_opd - act2_cca_opd.mean(axis=1, keepdims=True)

            act1_norm = act1_norm / np.linalg.norm(act1_norm)
            act2_norm = act2_norm / np.linalg.norm(act2_norm)

            _, cca_rho, _, transformed_rep1, _ = cca_decomp(act1_norm, act2_norm)
            val = pwcca_dist(act1_norm, cca_rho, transformed_rep1)

        elif metric == "procrustes":
            # normalize activations first

            act1_norm = act1_cca_opd - act1_cca_opd.mean(axis=1, keepdims=True)
            act2_norm = act2_cca_opd - act2_cca_opd.mean(axis=1, keepdims=True)

            act1_norm = act1_norm / np.linalg.norm(act1_norm)
            act2_norm = act2_norm / np.linalg.norm(act2_norm)
            val = procrustes(act1_norm, act2_norm)

        metric_values[metric] = val

    return metric_values
