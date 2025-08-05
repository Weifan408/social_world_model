import copy

import numpy as np
import torch
import torch.nn as nn


ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def inverse_symlog(y):
    return torch.sign(y) * (torch.exp(torch.abs(y)) - 1)


def two_hot(
    value,
    num_buckets: int = 255,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0,
    dtype=None,
):
    """Returns a two-hot vector of dim=num_buckets with two entries that are non-zero.

    See [1] for more details:
    [1] Mastering Diverse Domains through World Models - 2023
    D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
    https://arxiv.org/pdf/2301.04104v1.pdf

    Entries in the vector represent equally sized buckets within some fixed range
    (`lower_bound` to `upper_bound`).
    Those entries not 0.0 at positions k and k+1 encode the actual `value` and sum
    up to 1.0. They are the weights multiplied by the buckets values at k and k+1 for
    retrieving `value`.

    Example:
        num_buckets=11
        lower_bound=-5
        upper_bound=5
        value=2.5
        -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        -> [-5   -4   -3   -2   -1   0    1    2    3    4    5] (0.5*2 + 0.5*3=2.5)

    Example:
        num_buckets=5
        lower_bound=-1
        upper_bound=1
        value=0.1
        -> [0.0, 0.0, 0.8, 0.2, 0.0]
        -> [-1  -0.5   0   0.5   1] (0.2*0.5 + 0.8*0=0.1)

    Args:
        value: The input tensor of shape (B,) to be two-hot encoded.
        num_buckets: The number of buckets to two-hot encode into.
        lower_bound: The lower bound value used for the encoding. If input values are
            lower than this boundary, they will be encoded as `lower_bound`.
        upper_bound: The upper bound value used for the encoding. If input values are
            higher than this boundary, they will be encoded as `upper_bound`.

    Returns:
        The two-hot encoded tensor of shape (B, num_buckets).
    """
    # First make sure, values are clipped.
    value = torch.clamp(value, min=lower_bound, max=upper_bound)
    B = value.shape[0]
    device = value.device
    dtype = dtype or value.dtype
    # Tensor of batch indices: [0, B=batch size).

    bucket_delta = (upper_bound - lower_bound) / (num_buckets - 1)
    idx = (value - lower_bound) / bucket_delta

    k = idx.floor().long().clamp(0, num_buckets-1)  # (B,)
    kp1 = (k + 1).clamp(max=num_buckets-1)             # (B,)
    frac = (idx - k.to(idx.dtype)).clamp(0, 1)

    wkp1 = frac
    wk = 1.0 - frac
    out = torch.zeros(B, num_buckets, device=device, dtype=dtype)
    batch_idx = torch.arange(B, device=device)
    out[batch_idx, k] += wk
    out[batch_idx, kp1] += wkp1

    return out


def to_torch(obj, *, device, dtype=torch.float32):
    """
    Recursively move every ndarray / tensor in `obj` onto `device`
    and cast to `dtype` (only for floating types).

    Args
    ----
    obj : Arbitrary nested structure (dict / list / tuple / ndarray / tensor …)
    device : torch.device or str
    dtype : torch.dtype, only applied to float/complex arrays & tensors.

    Returns
    -------
    Same structure as `obj`, with arrays -> torch.Tensors on `device`.
    """

    if isinstance(obj, np.ndarray):
        ten = torch.as_tensor(obj)
        return _cast(ten, device, dtype)

    if isinstance(obj, torch.Tensor):
        return _cast(obj, device, dtype)

    if isinstance(obj, dict):
        return {k: to_torch(v, device=device, dtype=dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_torch(v, device=device, dtype=dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_torch(v, device=device, dtype=dtype) for v in obj)
    return obj


def _cast(tensor, device, dtype):
    tensor = tensor.to(device=device)
    if tensor.is_floating_point() or tensor.is_complex():
        tensor = tensor.to(dtype=dtype)
    return tensor
