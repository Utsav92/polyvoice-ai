import torch
from ml.models.fastspeech2.model import FastSpeech2, FS2Config, fs2_loss

def _batch(B=3, T=20, V=34):
    tok = torch.randint(1, V, (B, T)); tl = torch.tensor([20, 15, 18])
    dur = torch.randint(1, 6, (B, T)) * (torch.arange(T)[None] < tl[:, None]); ml = dur.sum(1)
    return tok, tl, dur, torch.randn(B, T), torch.randn(B, T), ml, torch.randn(B, int(ml.max()), 100)

def test_forward_loss_backward_and_inference_shapes():
    m = FastSpeech2(FS2Config(n_vocab=34, hidden=32, ffn=64, enc_layers=1, dec_layers=1))
    tok, tl, dur, p, e, ml, mel = _batch()
    o = m(tok, tl, dur, p, e, ml); loss, parts = fs2_loss(o, mel, dur, p, e); loss.backward()
    assert o["mel"].shape[:2] == (3, int(ml.max())) and o["mel"].shape[2] == 100 and torch.isfinite(loss)
    assert all(p.grad is not None for n, p in m.named_parameters() if "pos" not in n)
    m.eval(); o = m(tok[:1], tl[:1]); assert o["mel"].shape[0] == 1 and o["mel"].shape[1] == int(o["mel_len"]) >= 1

def test_length_regulator_expands_exactly():
    x = torch.arange(6.).view(1, 3, 2); out = FastSpeech2.regulate(x, torch.tensor([[2, 0, 3]]), torch.tensor([5]))
    assert out.shape == (1, 5, 2) and torch.equal(out[0, :, 0], torch.tensor([0., 0., 4., 4., 4.]))
