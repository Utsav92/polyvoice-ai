"""Monotonic Alignment Search (Glow-TTS / VITS), numba on CPU. Requires t_y >= t_x."""
import numpy as np
import torch
from numba import njit

NEG = -1e9

@njit(cache=True)
def _mas_one(value, t_y, t_x):
    path = np.zeros((t_y, t_x), dtype=np.int32)
    v = value.copy()
    for y in range(t_y):
        lo = max(0, t_x + y - t_y)
        hi = min(t_x, y + 1)
        for x in range(lo, hi):
            v_cur = NEG if x == y else v[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else NEG
            else:
                v_prev = v[y - 1, x - 1]
            v[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or v[y - 1, index] < v[y - 1, index - 1]):
            index -= 1
    return path

def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent [b,t_y,t_x] (higher=better), mask [b,t_y,t_x] -> hard path [b,t_y,t_x]."""
    device, dtype = neg_cent.device, neg_cent.dtype
    nc = neg_cent.detach().float().cpu().numpy()
    m = mask.detach().cpu().numpy()
    out = np.zeros(nc.shape, dtype=np.int32)
    for b in range(nc.shape[0]):
        t_y, t_x = int(m[b, :, 0].sum()), int(m[b, 0, :].sum())
        if t_y < t_x:
            raise ValueError(f"MAS needs frames >= tokens, got {t_y} < {t_x}")
        out[b, :t_y, :t_x] = _mas_one(nc[b, :t_y, :t_x].copy(), t_y, t_x)
    return torch.from_numpy(out).to(device=device, dtype=dtype)
