import math, torch
import torch.nn.functional as F

def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(k, d=1): return int((k * d - d) / 2)

def sequence_mask(length, max_length=None):
    if max_length is None: max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def fused_add_tanh_sigmoid_multiply(a, b, n_channels: int):
    s = a + b
    return torch.tanh(s[:, :n_channels]) * torch.sigmoid(s[:, n_channels:])

def generate_path(duration, mask):
    """duration [b,1,t_x], mask [b,1,t_y,t_x] -> hard monotonic alignment path [b,1,t_y,t_x]."""
    b, _, t_y, t_x = mask.shape
    cum = torch.cumsum(duration, -1).view(b * t_x)
    path = sequence_mask(cum, t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask

def rand_slice_segments(x, lengths, segment_size):
    b, _, t = x.shape
    max_start = (lengths - segment_size).clamp(min=0)
    starts = (torch.rand(b, device=x.device) * (max_start + 1).float()).long()
    segs = torch.stack([x[i, :, s:s + segment_size] for i, s in enumerate(starts.tolist())])
    return segs, starts

def slice_segments(x, starts, segment_size):
    return torch.stack([x[i, :, s:s + segment_size] for i, s in enumerate(starts.tolist())])

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)
