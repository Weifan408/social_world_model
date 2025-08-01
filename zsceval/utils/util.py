import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def convert_to_tensor(obj: Any) -> Any:
    """
    Recursively convert numpy.ndarray → torch.Tensor.
    Supports nested dict / list / tuple.
    Leaves torch.Tensor and non‑array types unchanged.
    """
    # 1. 已经是 Tensor，直接返回
    if isinstance(obj, torch.Tensor):
        return obj

    # 2. numpy 数组 → Tensor（自动共享内存，不拷贝）
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)

    # 3. 映射类型（dict、OrderedDict …）
    if isinstance(obj, Mapping):
        return {k: convert_to_tensor(v) for k, v in obj.items()}

    # 4. 序列类型（list、tuple …）
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        typ = type(obj)                          # 保留原容器类型
        return typ(convert_to_tensor(v) for v in obj)

    return obj


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
    else:
        return input


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == "Dict":
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def get_dim_from_act_space(action_space):
    if action_space.__class__.__name__ == "Discrete":
        action_dim = action_space.n
    elif action_space.__class__.__name__ == "Box":
        action_dim = action_space.shape[0]
    elif action_space.__class__.__name__ == "MultiBinary":
        action_dim = action_space.shape[0]
    elif action_space.__class__.__name__ == "MultiDiscrete":
        action_dim = action_space.high - action_space.low + 1
    return action_dim


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(
        list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c
