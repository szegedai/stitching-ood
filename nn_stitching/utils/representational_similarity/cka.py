from typing import Union, Optional

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from nn_stitching.utils import get_internal_activation
from nn_stitching.utils.low_rank import low_rank_approx


def compute_cka(model1: nn.Module,
                model2: nn.Module,
                layer1: str,
                layer2: str,
                data_loader: DataLoader,
                n_samples: int,
                device: Union[torch.device, str],
                cross_task: bool = False,
                model1_rank: Optional[int] = None,
                model2_rank: Optional[int] = None,
) -> float:

    act1 = []
    act2 = []

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

    return cka(act1.numpy(), act2.numpy())


def cka(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = gram_linear(rearrange_activations(x1))
    x2 = gram_linear(rearrange_activations(x2))
    similarity = _cka(x1, x2)
    return similarity


def rearrange_activations(activations: np.ndarray) -> np.ndarray:
    batch_size = activations.shape[0]
    flat_activations = activations.reshape(batch_size, -1)
    return flat_activations


def gram_linear(x: np.ndarray) -> np.ndarray:
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)

def center_gram(gram: np.ndarray, unbiased: bool = False) -> np.ndarray:
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=gram.dtype) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=gram.dtype)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram

def _cka(gram_x: np.ndarray, gram_y: np.ndarray, debiased: bool = False) -> float:
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


