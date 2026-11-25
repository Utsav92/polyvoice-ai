"""FastSpeech 2 (Ren et al. 2021), trained from scratch. Token-level pitch/energy, log-duration, Vocos-compatible mel output.
Pretrained component: none inside this module (the Vocos vocoder lives outside, see adapter)."""
from __future__ import annotations
from dataclasses import dataclass
import math, torch
from torch import nn
import torch.nn.functional as F
from ml.models.vits.commons import sequence_mask, generate_path

@dataclass
class FS2Config:
    n_vocab: int = 64
    n_mels: int = 100
    hidden: int = 256
    heads: int = 2
    enc_layers: int = 4
    dec_layers: int = 4
    ffn: int = 512
    ffn_kernel: int = 9
    dropout: float = 0.1
    var_filter: int = 256
    var_kernel: int = 3
    n_bins: int = 256
    pitch_min: float = -3.0      # in standardized (z-score of log-F0 over voiced frames) units; set from data stats
    pitch_max: float = 3.0
    energy_min: float = -3.0
    energy_max: float = 3.0
    n_speakers: int = 1
    n_languages: int = 1

class PosEnc(nn.Module):
    def __init__(self, d, max_len=2000):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1); div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x): return x + self.pe[: x.size(1)].unsqueeze(0)

class FFTBlock(nn.Module):
    def __init__(self, d, heads, ffn, kernel, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, ffn, kernel, padding=kernel // 2); self.c2 = nn.Conv1d(ffn, d, 1); self.drop = nn.Dropout(p)
    def forward(self, x, pad_mask):  # x [b,t,d]; pad_mask [b,t] True where padding
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.n1(x + self.drop(a)); x = x.masked_fill(pad_mask.unsqueeze(-1), 0)
        f = self.c2(self.drop(torch.relu(self.c1(x.transpose(1, 2))))).transpose(1, 2)
        x = self.n2(x + self.drop(f)); return x.masked_fill(pad_mask.unsqueeze(-1), 0)

class VariancePredictor(nn.Module):
    def __init__(self, d, filt, kernel, p):
        super().__init__()
        self.c1 = nn.Conv1d(d, filt, kernel, padding=kernel // 2); self.c2 = nn.Conv1d(filt, filt, kernel, padding=kernel // 2)
        self.n1, self.n2 = nn.LayerNorm(filt), nn.LayerNorm(filt); self.out = nn.Linear(filt, 1); self.drop = nn.Dropout(p)
    def forward(self, x, pad_mask):
        h = self.drop(self.n1(torch.relu(self.c1(x.transpose(1, 2))).transpose(1, 2)))
        h = self.drop(self.n2(torch.relu(self.c2(h.transpose(1, 2))).transpose(1, 2)))
        return self.out(h).squeeze(-1).masked_fill(pad_mask, 0)

class FastSpeech2(nn.Module):
    def __init__(self, c: FS2Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.n_vocab, c.hidden)   # id 0 is the learnable silence token (padding is masked by length)
        self.pos = PosEnc(c.hidden)
        if c.n_speakers > 1: self.spk = nn.Embedding(c.n_speakers, c.hidden)
        if c.n_languages > 1: self.lang = nn.Embedding(c.n_languages, c.hidden)
        self.enc = nn.ModuleList([FFTBlock(c.hidden, c.heads, c.ffn, c.ffn_kernel, c.dropout) for _ in range(c.enc_layers)])
        self.dec = nn.ModuleList([FFTBlock(c.hidden, c.heads, c.ffn, c.ffn_kernel, c.dropout) for _ in range(c.dec_layers)])
        self.dur, self.pitch, self.energy = (VariancePredictor(c.hidden, c.var_filter, c.var_kernel, c.dropout) for _ in range(3))
        self.register_buffer("pitch_bins", torch.linspace(c.pitch_min, c.pitch_max, c.n_bins - 1), persistent=False)
        self.register_buffer("energy_bins", torch.linspace(c.energy_min, c.energy_max, c.n_bins - 1), persistent=False)
        self.pitch_emb, self.energy_emb = nn.Embedding(c.n_bins, c.hidden), nn.Embedding(c.n_bins, c.hidden)
        self.mel = nn.Linear(c.hidden, c.n_mels)

    def _cond(self, x, sid, lid):
        if sid is not None and hasattr(self, "spk"): x = x + self.spk(sid).unsqueeze(1)
        if lid is not None and hasattr(self, "lang"): x = x + self.lang(lid).unsqueeze(1)
        return x

    @staticmethod
    def regulate(x, dur, y_len):
        """Expand [b,t_x,d] by integer durations [b,t_x] into [b,t_y,d] (vectorised, via the VITS path generator)."""
        t_y = int(y_len.max())
        b, t_x = dur.shape
        mask = torch.ones(b, 1, t_y, t_x, device=x.device)
        path = generate_path(dur.unsqueeze(1).float(), mask).squeeze(1)       # [b,t_y,t_x]
        out = torch.matmul(path, x)
        keep = sequence_mask(y_len, t_y).unsqueeze(-1)
        return out * keep

    def forward(self, tokens, tok_len, dur=None, pitch=None, energy=None, mel_len=None, sid=None, lid=None,
                pitch_scale=1.0, energy_scale=1.0, speed=1.0):
        """Training: pass ground-truth dur/pitch/energy/mel_len (teacher forcing). Inference: leave None."""
        pad = ~sequence_mask(tok_len, tokens.size(1))
        x = self._cond(self.pos(self.emb(tokens) * math.sqrt(self.c.hidden)), sid, lid)
        for l in self.enc: x = l(x, pad)
        logd = self.dur(x, pad)      # predicts log(1+d), the standard FS2 target
        p_hat, e_hat = self.pitch(x, pad), self.energy(x, pad)
        if dur is None:
            dur_i = torch.clamp(torch.round((torch.exp(logd) - 1) / speed), min=0).long().masked_fill(pad, 0)
            dur_i = torch.where(dur_i.sum(1, keepdim=True) == 0, torch.ones_like(dur_i), dur_i).masked_fill(pad, 0)
            mel_len = dur_i.sum(1); p_use, e_use = p_hat * pitch_scale, e_hat * energy_scale
        else:
            dur_i, p_use, e_use = dur.long(), pitch, energy
        x = x + self.pitch_emb(torch.bucketize(p_use, self.pitch_bins)) + self.energy_emb(torch.bucketize(e_use, self.energy_bins))
        y = self.regulate(x, dur_i, mel_len)
        y_pad = ~sequence_mask(mel_len, y.size(1))
        y = self.pos(y)
        for l in self.dec: y = l(y, y_pad)
        return {"mel": self.mel(y), "logd": logd, "pitch": p_hat, "energy": e_hat, "pad": pad, "mel_pad": y_pad, "mel_len": mel_len, "dur": dur_i}

def fs2_loss(out, mel, dur, pitch, energy):
    keep, mkeep = ~out["pad"], ~out["mel_pad"]
    T = min(mel.size(1), out["mel"].size(1))
    m = mkeep[:, :T].unsqueeze(-1).float()
    l_mel = (torch.abs(out["mel"][:, :T] - mel[:, :T]) * m).sum() / (m.sum() * mel.size(-1))
    k = keep.float()
    l_dur = (((out["logd"] - torch.log1p(dur.float())) ** 2) * k).sum() / k.sum()
    l_p = (((out["pitch"] - pitch) ** 2) * k).sum() / k.sum(); l_e = (((out["energy"] - energy) ** 2) * k).sum() / k.sum()
    return l_mel + l_dur + l_p + l_e, {"mel": float(l_mel), "dur": float(l_dur), "pitch": float(l_p), "energy": float(l_e)}
