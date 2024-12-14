from typing import List, Optional

import torch

from nn_stitching.stitchers import Stitcher


def stitcher_l1_loss(x: torch.Tensor,
                     y: torch.Tensor,
                     stitcher: Stitcher,
                     coef: float = 1e-3):
    l1_penalty = torch.norm(stitcher.stitching_layer.transform.weight, 1) + \
                 torch.norm(stitcher.stitching_layer.transform.bias, 1)

    loss = torch.nn.CrossEntropyLoss()(x, y) + coef * l1_penalty

    return loss


def sparsity(weights: List[torch.Tensor],
             threshold : Optional[float] = 0
) -> float:
    n_elem = 0
    n_sparse = 0

    for weight in weights:
        n_elem += weight.numel()
        n_sparse += torch.sum((torch.abs(weight) <= threshold)).int().item()

    return n_sparse / n_elem
