import numpy as np


def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B

    Implementation taken directly from Ding et al. Grounding Representation
    Similarity with Statistical Testing (NeurIPS 2021)
    """
    A_sq_frob = np.sum(A ** 2)
    B_sq_frob = np.sum(B ** 2)
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc
