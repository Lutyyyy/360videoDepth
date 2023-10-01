import cv2
import numpy as np
from PIL import Image

import torchvision


def detach_to_cpu(tensor):
    if type(tensor) == np.ndarray:
        return tensor
    else:
        if tensor.requires_grad:
            tensor.requires_grad = False
        tensor = tensor.cpu()
    return tensor.numpy()


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = detach_to_cpu(x)
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_

def depth2img(tensor, normalize=True, disparity=True, eps=1e-6, **kargs):
    t = detach_to_cpu(tensor)
    assert len(t.shape) == 4
    assert t.shape[1] == 1
    t = 1 / (t + eps)
    # if normalize:
    max_v = np.max(t, axis=(2, 3), keepdims=True)
    min_v = np.min(t, axis=(2, 3), keepdims=True)
    t = (t - min_v) / (max_v - min_v + eps)
    #    return t
    # else:
    #    return t
    cs = []
    for b in range(t.shape[0]):
        c = heatmap_to_pseudo_color(t[b, 0, ...])
        cs.append(c[None, ...])
    cs = np.concatenate(cs, axis=0)
    cs = np.transpose(cs, [0, 3, 1, 2])
    return cs
