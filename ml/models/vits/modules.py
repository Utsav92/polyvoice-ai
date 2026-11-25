import math, torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from .commons import init_weights, get_padding, fused_add_tanh_sigmoid_multiply, sequence_mask

LRELU = 0.1

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__(); self.g = nn.Parameter(torch.ones(channels)); self.b = nn.Parameter(torch.zeros(channels)); self.eps = eps
    def forward(self, x):  # [b,c,t]
        return F.layer_norm(x.transpose(1, -1), (x.size(1),), self.g, self.b, self.eps).transpose(1, -1)

class WN(nn.Module):
    """WaveNet-style gated dilated conv stack with optional global conditioning g."""
    def __init__(self, hidden, kernel, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.in_layers, self.res_skip_layers = nn.ModuleList(), nn.ModuleList()
        if gin_channels: self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden * n_layers, 1))
        for i in range(n_layers):
            d = dilation_rate ** i
            self.in_layers.append(weight_norm(nn.Conv1d(hidden, 2 * hidden, kernel, dilation=d, padding=int((kernel * d - d) / 2))))
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden, 2 * hidden if i < n_layers - 1 else hidden, 1)))
        self.gin = gin_channels
    def forward(self, x, x_mask, g=None):
        out = torch.zeros_like(x)
        if g is not None and self.gin: g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * self.hidden:(i + 1) * 2 * self.hidden] if (g is not None and self.gin) else torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden)
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + rs[:, :self.hidden]) * x_mask; out = out + rs[:, self.hidden:]
            else:
                out = out + rs
        return out * x_mask

class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden, kernel, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, kernel, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden, self.half, 1)
        self.post.weight.data.zero_(); self.post.bias.data.zero_()
    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half, self.half], 1)
        m = self.post(self.enc(self.pre(x0) * x_mask, x_mask, g)) * x_mask
        if not reverse:
            x1 = m + x1 * x_mask; return torch.cat([x0, x1], 1), torch.zeros(x.size(0), device=x.device)
        x1 = (x1 - m) * x_mask; return torch.cat([x0, x1], 1)

class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden, kernel, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList([ResidualCouplingLayer(channels, hidden, kernel, dilation_rate, n_layers, gin_channels) for _ in range(n_flows)])
    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for f in self.flows:
                x, _ = f(x, x_mask, g); x = torch.flip(x, [1])
        else:
            for f in reversed(self.flows):
                x = torch.flip(x, [1]); x = f(x, x_mask, g, reverse=True)
        return x

class PosteriorEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, kernel, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_ch = out_ch
        self.pre = nn.Conv1d(in_ch, hidden, 1); self.enc = WN(hidden, kernel, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden, out_ch * 2, 1)
    def forward(self, x, x_lengths, g=None):
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        stats = self.proj(self.enc(self.pre(x) * x_mask, x_mask, g)) * x_mask
        m, logs = torch.split(stats, self.out_ch, 1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class TransformerLayer(nn.Module):
    def __init__(self, hidden, filt, heads, kernel=3, p=0.1, rel_window=0):
        super().__init__()
        self.h, self.heads, self.rel_window = hidden, heads, rel_window
        if rel_window: self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * rel_window + 1))   # learned per-head bias on clipped relative distance j-i
        self.qkv = nn.Conv1d(hidden, 3 * hidden, 1); self.o = nn.Conv1d(hidden, hidden, 1)
        self.n1, self.n2 = LayerNorm(hidden), LayerNorm(hidden)
        self.c1 = nn.Conv1d(hidden, filt, kernel, padding=kernel // 2); self.c2 = nn.Conv1d(filt, hidden, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(p)
    def forward(self, x, x_mask):
        b, c, t = x.shape
        q, k, v = self.qkv(x).chunk(3, 1)
        sp = lambda a: a.view(b, self.heads, c // self.heads, t).transpose(2, 3)
        attn_mask = (x_mask.unsqueeze(2) * x_mask.unsqueeze(3)).bool()  # [b,1,t,t]
        if self.rel_window:
            pos = torch.arange(t, device=x.device); rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-self.rel_window, self.rel_window) + self.rel_window
            bias = self.rel_bias[:, rel].unsqueeze(0).to(q.dtype)                       # [1,h,t,t]
            attn_mask = bias.masked_fill(~attn_mask, -1e4)                               # float mask; -1e4 (not -inf) so fully padded rows stay finite
        y = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), attn_mask=attn_mask)
        y = y.transpose(2, 3).reshape(b, c, t)
        x = self.n1(x + self.drop(self.o(y)))
        f = self.c2(self.drop(torch.relu(self.c1(x * x_mask))) * x_mask) * x_mask
        return self.n2(x + self.drop(f))

class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_ch, hidden, filt, heads, n_layers, kernel=3, p=0.1, rel_window=0):
        super().__init__()
        self.hidden, self.out_ch, self.rel_window = hidden, out_ch, rel_window
        self.emb = nn.Embedding(n_vocab, hidden); nn.init.normal_(self.emb.weight, 0.0, hidden ** -0.5)
        self.layers = nn.ModuleList([TransformerLayer(hidden, filt, heads, kernel, p, rel_window) for _ in range(n_layers)])
        self.proj = nn.Conv1d(hidden, out_ch * 2, 1)
    @staticmethod
    def pos_enc(t, c, device):
        pos = torch.arange(t, device=device).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2, device=device).float() * (-math.log(10000.0) / c))
        pe = torch.zeros(t, c, device=device); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe.t().unsqueeze(0)
    def forward(self, x, x_lengths):
        x_mask = sequence_mask(x_lengths, x.size(1)).unsqueeze(1).to(torch.float32)
        h = (self.emb(x) * math.sqrt(self.hidden)).transpose(1, 2)
        h = (h if self.rel_window else h + self.pos_enc(h.size(2), self.hidden, h.device)) * x_mask
        for l in self.layers: h = l(h, x_mask) * x_mask
        stats = self.proj(h) * x_mask
        m, logs = torch.split(stats, self.out_ch, 1)
        return h, m, logs, x_mask

class DurationPredictor(nn.Module):
    """Deterministic duration predictor (log-domain), trained against MAS durations."""
    def __init__(self, in_ch, filt, kernel=3, p=0.5, gin_channels=0):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.c1 = nn.Conv1d(in_ch, filt, kernel, padding=kernel // 2); self.n1 = LayerNorm(filt)
        self.c2 = nn.Conv1d(filt, filt, kernel, padding=kernel // 2); self.n2 = LayerNorm(filt)
        self.proj = nn.Conv1d(filt, 1, 1)
        if gin_channels: self.cond = nn.Conv1d(gin_channels, in_ch, 1)
    def forward(self, x, x_mask, g=None):
        x = x.detach()
        if g is not None and hasattr(self, "cond"): x = x + self.cond(g.detach())
        x = self.drop(self.n1(torch.relu(self.c1(x * x_mask)))); x = self.drop(self.n2(torch.relu(self.c2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask

class ResBlock1(nn.Module):
    def __init__(self, ch, kernel=3, dilation=(1, 3, 5)):
        super().__init__()
        self.c1 = nn.ModuleList([weight_norm(Conv1d(ch, ch, kernel, 1, dilation=d, padding=get_padding(kernel, d))) for d in dilation])
        self.c2 = nn.ModuleList([weight_norm(Conv1d(ch, ch, kernel, 1, dilation=1, padding=get_padding(kernel, 1))) for _ in dilation])
        for c in list(self.c1) + list(self.c2): c.apply(init_weights)
    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = b(F.leaky_relu(a(F.leaky_relu(x, LRELU)), LRELU)) + x
        return x

class Generator(nn.Module):
    """HiFi-GAN V1-style decoder, scaled by upsample_initial_channel."""
    def __init__(self, in_ch, init_ch, rb_kernels, rb_dilations, up_rates, up_kernels, gin_channels=0):
        super().__init__()
        self.nk = len(rb_kernels)
        self.pre = Conv1d(in_ch, init_ch, 7, 1, padding=3)
        self.ups = nn.ModuleList(); self.rbs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(up_rates, up_kernels)):
            self.ups.append(weight_norm(ConvTranspose1d(init_ch // 2 ** i, init_ch // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
            ch = init_ch // 2 ** (i + 1)
            for k2, d in zip(rb_kernels, rb_dilations): self.rbs.append(ResBlock1(ch, k2, d))
        self.post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        if gin_channels: self.cond = nn.Conv1d(gin_channels, init_ch, 1)
    def forward(self, x, g=None):
        x = self.pre(x)
        if g is not None and hasattr(self, "cond"): x = x + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = None
            for j in range(self.nk):
                y = self.rbs[i * self.nk + j](x); xs = y if xs is None else xs + y
            x = xs / self.nk
        return torch.tanh(self.post(F.leaky_relu(x)))

class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel=5, stride=3, ch=(32, 128, 512, 1024)):
        super().__init__()
        self.period = period
        chans = [1, *ch]
        self.convs = nn.ModuleList([weight_norm(Conv2d(chans[i], chans[i + 1], (kernel, 1), (stride, 1), padding=(get_padding(kernel, 1), 0))) for i in range(len(ch))])
        self.convs.append(weight_norm(Conv2d(ch[-1], ch[-1], (kernel, 1), 1, padding=(2, 0))))
        self.post = weight_norm(Conv2d(ch[-1], 1, (3, 1), 1, padding=(1, 0)))
    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period: x = F.pad(x, (0, self.period - t % self.period), "reflect"); t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs: x = F.leaky_relu(l(x), LRELU); fmap.append(x)
        x = self.post(x); fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral=False):
        super().__init__()
        n = spectral_norm if use_spectral else weight_norm
        self.convs = nn.ModuleList([n(Conv1d(1, 16, 15, 1, padding=7)), n(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            n(Conv1d(64, 256, 41, 4, groups=16, padding=20)), n(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            n(Conv1d(512, 512, 41, 4, groups=16, padding=20)), n(Conv1d(512, 512, 5, 1, padding=2))])
        self.post = n(Conv1d(512, 1, 3, 1, padding=1))
    def forward(self, x):
        fmap = []
        for l in self.convs: x = F.leaky_relu(l(x), LRELU); fmap.append(x)
        x = self.post(x); fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.ds = nn.ModuleList([DiscriminatorS(True)] + [DiscriminatorP(p) for p in periods])
    def forward(self, y, y_hat):
        r, g, fr, fg = [], [], [], []
        for d in self.ds:
            a, fa = d(y); b, fb = d(y_hat); r.append(a); g.append(b); fr.append(fa); fg.append(fb)
        return r, g, fr, fg
