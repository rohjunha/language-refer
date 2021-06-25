import torch
from torch import Tensor


def pe_from_tensor(
        src: Tensor,
        n_dim: int,
        dim: int = -1,
        collapse: bool = True) -> Tensor:
    """
    Positional encoding from raw input
    :param src: (..., in_dim), Tensor
    :param n_dim: output dimension, int
    :param dim: dimension to apply PE
    :param collapse: keeps the dimensions of the output tensor
    :return: positional encoded input tensor
    """
    dims = [1] * (src.ndim + 1)
    dims[dim] = n_dim
    den = src.unsqueeze(dim).repeat(*dims)
    nom = torch.tensor([torch.pow(torch.tensor(10000.), 2 * (j // 2) / n_dim) for j in range(n_dim)],
                       device=src.device).view(*dims)
    res = torch.div(den, nom)
    if collapse:
        shapes = list(src.shape)
        shapes[-1] = -1
        res = res.view(shapes)
    assert res.device == src.device
    return res
