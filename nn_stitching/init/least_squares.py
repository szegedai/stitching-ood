from typing import Optional, Tuple

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from general_utils.datasets import DATASET_META
from general_utils.datasets.imagenet import IMAGENET_VAL_TRANSFORM
from general_utils.general import get_layer

from nn_stitching.stitchers import Stitcher
from nn_stitching.models import StitchedModel
from nn_stitching.utils.low_rank import low_rank_approx


def least_squares_init(front_model: Stitcher,
                       end_model: StitchedModel,
                       front_layer_name: str,
                       end_layer_name: str,
                       dataset_name: str,
                       n_samples: int,
                       stitcher: Stitcher,
                       dataset_root: Optional[str] = "./data/pytorch",
                       front_pre_activation: Optional[bool] = False,
                       end_pre_activation: Optional[bool] = False,
                       front_rank: Optional[int] = None
) -> Stitcher:
    r"""Initializes the stitching layer's parameters using the least-squares
    matching method.

    """

    front_layer = get_layer(front_model, front_layer_name)
    end_layer = get_layer(end_model, end_layer_name)

    front_activations = None
    end_activations = None

    def front_activation_store_hook(module, m_in, m_out):
        nonlocal front_activations
        nonlocal front_pre_activation
        if front_pre_activation:
            front_activations = m_in[0].detach().cpu()
        else:
            front_activations = m_out.detach().cpu()

    def end_activation_store_hook(module, m_in, m_out):
        nonlocal end_activations
        nonlocal end_pre_activation
        if end_pre_activation:
            end_activations = m_in[0].detach().cpu()
        else:
            end_activations = m_out.detach().cpu()

    front_hook_handler = front_layer.register_forward_hook(front_activation_store_hook)
    end_hook_handler = end_layer.register_forward_hook(end_activation_store_hook)

    front_model.to("cpu")
    end_model.to("cpu")
    front_model.eval()
    end_model.eval()

    dataset_func = DATASET_META[dataset_name]["dataset_func"]
    dataset = dataset_func(root=dataset_root, train=True, download=True, transform=transforms.ToTensor()
                                                                         if "imagenet" not in dataset_name
                                                                         else IMAGENET_VAL_TRANSFORM)
    data_loader = DataLoader(dataset, batch_size=n_samples, shuffle=True, num_workers=16, prefetch_factor=8)

    data = next(iter(data_loader))
    x = data[0].to("cpu")
    front_model(x)
    end_model(x)

    front_hook_handler.remove()
    end_hook_handler.remove()

    if front_rank:
        front_activations = low_rank_approx(front_activations, front_rank)

    pinv_w_b = ps_inv(front_activations, end_activations)
    pinv_loss = ps_inv_loss(front_activations, end_activations)
    pinv_w = pinv_w_b["w"]
    pinv_b = pinv_w_b["b"]

    weight_shape = stitcher.stitching_layer.transform.weight.shape
    bias_shape = stitcher.stitching_layer.transform.bias.shape

    weight = torch.Tensor(pinv_w).reshape(weight_shape)
    bias = torch.Tensor(pinv_b).reshape(bias_shape)

    stitcher.stitching_layer.transform.weight = nn.Parameter(weight)
    stitcher.stitching_layer.transform.bias = nn.Parameter(bias)

    return stitcher, pinv_loss.item()


#===============================================================================
# The below code is taken directly from Csiszárik et al.
# Similarity and Matching of Neural Network Representations (NeurIPS 2021)
# with the following modifications:
# - torch pinv instead of numpy
# - works for transformer embeddings
#===============================================================================

def ps_inv(x1, x2):

    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when '\
                            'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = torch.ones(shape).to(x1.device)
    x1_ones[:, :-1] = x1
    A_ones = torch.matmul(torch.linalg.pinv(x1_ones), x2).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    return {'w' : w, 'b' : b}


# TODO: make part of least_squares_init
def ps_inv_loss(x1, x2):

    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)

    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when '\
                            'calculating psuedo inverse matrix.')

    # Get transformation matrix shape
    shape = list(x1.shape)
    shape[-1] += 1

    # Calculate pseudo inverse
    x1_ones = torch.ones(shape).to(x1.device)
    x1_ones[:, :-1] = x1
    A_ones = torch.matmul(torch.linalg.pinv(x1_ones, atol=0, rtol=0), x2).T

    # Get weights and bias
    w = A_ones[..., :-1]
    b = A_ones[..., -1]

    matched_x2 = torch.matmul(x1, w.T) + b
    match_diff = matched_x2 - x2

    return (torch.norm(match_diff, "fro") ** 2) / (torch.norm(x2, "fro") ** 2)


def rearrange_activations(activations):
    is_convolution = len(activations.shape) == 4
    is_trans = len(activations.shape) == 3
    if is_convolution:
        # activations = np.transpose(activations, axes=[0, 2, 3, 1])
        activations = activations.permute(0, 2, 3, 1)
        n_channels = activations.shape[-1]
        new_shape = (-1, n_channels)
    elif is_trans:
        embed_dim = activations.shape[-1]
        new_shape = (-1, embed_dim)
    else:
        new_shape = (activations.shape[0], -1)

    reshaped_activations = activations.reshape(*new_shape)
    return reshaped_activations
