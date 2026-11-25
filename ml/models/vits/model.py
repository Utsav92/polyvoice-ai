"""Scaled-down VITS (Kim et al. 2021): posterior encoder, normalizing flow, MAS alignment, deterministic duration predictor, HiFi-GAN decoder."""
from __future__ import annotations
from dataclasses import dataclass, field
import math, torch
from torch import nn
import torch.nn.functional as F
from .commons import sequence_mask, generate_path, rand_slice_segments, kl_loss
from .mas import maximum_path
from . import modules as M

@dataclass
class VitsConfig:
    n_vocab: int = 64
    sample_rate: int = 16000
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 80
    inter_channels: int = 192
    hidden_channels: int = 128
    filter_channels: int = 512
    n_heads: int = 2
    n_layers: int = 4
    p_dropout: float = 0.1
    resblock_kernels: tuple = (3, 7, 11)
    resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple = (8, 8, 2, 2)           # product must equal hop
    upsample_kernels: tuple = (16, 16, 4, 4)
    upsample_initial_channel: int = 256
    segment_size: int = 8192
    n_speakers: int = 1
    n_languages: int = 1
    gin_channels: int = 64
    wn_layers_post: int = 8
    flow_layers: int = 4
    rel_pos_window: int = 0     # >0: relative-position attention bias in the text encoder (replaces absolute sinusoids)

    def __post_init__(self):
        p = 1
        for u in self.upsample_rates: p *= u
        assert p == self.hop, f"upsample product {p} != hop {self.hop}"

class VITS(nn.Module):
    def __init__(self, c: VitsConfig):
        super().__init__()
        self.c = c
        g = c.gin_channels if (c.n_speakers > 1 or c.n_languages > 1) else 0
        self.g_ch = g
        if c.n_speakers > 1: self.emb_spk = nn.Embedding(c.n_speakers, c.gin_channels)
        if c.n_languages > 1: self.emb_lang = nn.Embedding(c.n_languages, c.gin_channels)
        self.enc_p = M.TextEncoder(c.n_vocab, c.inter_channels, c.hidden_channels, c.filter_channels, c.n_heads, c.n_layers, 3, c.p_dropout, c.rel_pos_window)
        self.enc_q = M.PosteriorEncoder(c.n_fft // 2 + 1, c.inter_channels, c.hidden_channels, 5, 1, c.wn_layers_post, g)
        self.flow = M.ResidualCouplingBlock(c.inter_channels, c.hidden_channels, 5, 1, 4, c.flow_layers, g)
        self.dec = M.Generator(c.inter_channels, c.upsample_initial_channel, c.resblock_kernels, c.resblock_dilations, c.upsample_rates, c.upsample_kernels, g)
        self.dp = M.DurationPredictor(c.hidden_channels, 256, 3, 0.5, g)

    def _g(self, sid, lid):
        if not self.g_ch: return None
        g = 0
        if self.c.n_speakers > 1: g = g + self.emb_spk(sid)
        if self.c.n_languages > 1: g = g + self.emb_lang(lid)
        return g.unsqueeze(-1)

    def forward(self, x, x_len, spec, spec_len, sid=None, lid=None, ext_dur=None):
        """ext_dur [b,t_x] (frames): if given, use this hard alignment instead of MAS (alignment bootstrap)."""
        g = self._g(sid, lid)
        x_h, m_p, logs_p, x_mask = self.enc_p(x, x_len)
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_len, g)
        z_p = self.flow(z, y_mask, g)
        with torch.no_grad(), torch.autocast(device_type=z_p.device.type, enabled=False):
            z_p, m_p, logs_p = z_p.float(), m_p.float(), logs_p.float()
            s_p_sq_r = torch.exp(-2 * logs_p)
            nc1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            nc2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            nc3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            nc4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = nc1 + nc2 - nc3 + nc4
            attn_mask = (y_mask.unsqueeze(-1) * x_mask.unsqueeze(2))     # [b,1,t_y,t_x]
            if ext_dur is None:
                attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
            else:
                attn = generate_path(ext_dur.unsqueeze(1).to(attn_mask.dtype), attn_mask).detach()
        w = attn.sum(2)                                                   # [b,1,t_x] durations
        logw_hat = self.dp(x_h, x_mask, g)
        with torch.no_grad():   # alignment diagnostics
            self.last_diag = {"dur_mae_frames": float((torch.exp(logw_hat) - w).abs().mul(x_mask).sum() / x_mask.sum()),
                              "mas_ll": float((neg_cent * attn.squeeze(1)).sum() / y_mask.sum())}
        logw = torch.log(w + 1e-6) * x_mask
        l_dur = torch.sum((logw_hat - logw) ** 2 * x_mask) / torch.sum(x_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_slice, starts = rand_slice_segments(z, spec_len, self.c.segment_size // self.c.hop)
        o = self.dec(z_slice, g)
        l_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        return o, l_dur, l_kl, attn, starts, (x_mask, y_mask)

    @torch.no_grad()
    def align(self, x, x_len, spec, spec_len, sid=None, lid=None):
        """Teacher alignment: MAS durations [b,t_x] (in this model's frames) using the posterior MEAN (deterministic)."""
        g = self._g(sid, lid)
        _, m_p, logs_p, x_mask = self.enc_p(x, x_len)
        _, m_q, _, y_mask = self.enc_q(spec, spec_len, g)
        z_p = self.flow(m_q * y_mask, y_mask, g)
        s_p_sq_r = torch.exp(-2 * logs_p)
        nc = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
              - torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r) + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
        attn = maximum_path(nc, (y_mask.unsqueeze(-1) * x_mask.unsqueeze(2)).squeeze(1))
        return attn.sum(1)

    @torch.no_grad()
    def infer(self, x, x_len, sid=None, lid=None, noise_scale=0.667, length_scale=1.0, max_len=None):
        g = self._g(sid, lid)
        x_h, m_p, logs_p, x_mask = self.enc_p(x, x_len)
        logw = self.dp(x_h, x_mask, g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_len = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_len, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g)
        return o, attn, y_mask

def feature_loss(fr, fg):
    return sum(torch.mean(torch.abs(a.detach() - b)) for dr, dg in zip(fr, fg) for a, b in zip(dr, dg)) * 2

def discriminator_loss(r, g):
    return sum(torch.mean((1 - a) ** 2) + torch.mean(b ** 2) for a, b in zip(r, g))

def generator_loss(g):
    return sum(torch.mean((1 - b) ** 2) for b in g)
