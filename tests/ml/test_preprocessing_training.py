"""Unit tests on synthetic data (no downloads). The training test runs the REAL trainer for a few steps on a tiny synthetic corpus."""
import json, numpy as np, pytest, soundfile as sf, torch, yaml
from pathlib import Path
from ml.preprocessing.audio import load_validate, AudioRejected, to_int16
from ml.preprocessing.tokenizer import english_tokenizer
from ml.datasets.ljspeech import split_of
from ml.models.vits.mas import maximum_path

def _wav(path, secs=2.0, sr=22050, amp=0.3, freq=220.0):
    t = np.arange(int(secs * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32"), sr)

def test_decode_resample_and_mono(tmp_path):
    _wav(tmp_path / "a.wav", 2.0, 22050)
    y, rep = load_validate(tmp_path / "a.wav", 16000)
    assert abs(len(y) / 16000 - 2.0) < 0.01 and y.ndim == 1 and rep.sr_in == 22050

def test_rejections(tmp_path):
    _wav(tmp_path / "short.wav", 0.3); _wav(tmp_path / "loud.wav", 2.0, amp=5.0)
    sf.write(tmp_path / "silent.wav", np.zeros(32000, dtype="float32"), 16000)
    (tmp_path / "corrupt.wav").write_bytes(b"RIFFnotreallyawav")
    for name in ["short", "loud", "silent", "corrupt"]:
        with pytest.raises(AudioRejected): load_validate(tmp_path / f"{name}.wav")

def test_split_deterministic_and_proportions():
    ids = [f"LJ{i:03d}-{j:04d}" for i in range(1, 51) for j in range(1, 263)]
    s = [split_of(i) for i in ids]
    assert s == [split_of(i) for i in ids]
    frac_heldout = 1 - s.count("train") / len(s)
    assert 0.02 < frac_heldout < 0.08

def test_tokenizer_blank_and_normalization():
    tok = english_tokenizer(); ids = tok.encode("Hi, 3!", "en")
    assert ids[0] == 0 and ids[-1] == 0 and all(i == 0 for i in ids[0::2])
    assert tok.symbols[ids[1]] == "h"

def test_mas_monotonic_and_covers_all_tokens():
    torch.manual_seed(0); p = maximum_path(torch.randn(2, 30, 10), torch.ones(2, 30, 10))
    for b in range(2):
        assert (p[b].sum(1) == 1).all() and (p[b].sum(0) >= 1).all()
        cols = p[b].argmax(1); assert (cols[1:] >= cols[:-1]).all()

def _tiny_corpus(root: Path, n=24):
    (root / "wavs16k").mkdir(parents=True)
    tok = english_tokenizer(); tok.save(root / "tokenizer.json"); rows = []
    for i in range(n):
        y = np.sin(2 * np.pi * (180 + 10 * i) * np.arange(16000 * 2) / 16000).astype("float32") * 0.3
        np.save(root / "wavs16k" / f"u{i}.npy", to_int16(y))
        txt = "hello there"; rows.append({"id": f"u{i}", "text_norm": txt, "duration_s": 2.0, "tokens": tok.encode(txt), "language": "en", "speaker": "s", "split": "val" if i < 4 else "train"})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

@pytest.mark.real_model
def test_training_checkpoint_resume(tmp_path):
    from ml.training.vits_trainer import Trainer
    _tiny_corpus(tmp_path / "d")
    cfg = yaml.safe_load(open("configs/local_mps.yaml"))
    cfg.update(data_root=str(tmp_path / "d"), run_dir=str(tmp_path / "run"), device="cpu")
    cfg["model"].update(hidden_channels=32, filter_channels=64, inter_channels=32, upsample_initial_channel=32, n_layers=1)
    cfg["train"].update(batch_size=4, workers=0, max_steps=3, log_every=1, val_every=100, ckpt_every=2, gan_start_step=1)
    Trainer(cfg, resume=False).fit()
    log = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in log if e["event"] == "train"] == [1, 2, 3] and any(e["event"] == "ckpt" and e["size_mb"] > 0 for e in log)
    t2 = Trainer(cfg, resume=True)
    assert t2.step == 3                       # resumed from the final checkpoint
    w0 = {k: v.clone() for k, v in t2.G.state_dict().items()}
    cfg["train"]["max_steps"] = 5; t2.cfg = cfg; t2.fit()
    assert t2.step == 5 and any(not torch.equal(w0[k], v) for k, v in t2.G.state_dict().items())

def _two_language_corpus(root: Path, n=24):
    _tiny_corpus(root, n)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    for i, r in enumerate(rows): r["language"] = "en" if i % 3 else "xx"; r["speaker"] = "s1" if i % 2 else "s2"
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

def _tiny_cfg(tmp_path, **train):
    cfg = yaml.safe_load(open("configs/local_mps.yaml"))
    cfg.update(data_root=str(tmp_path / "d"), run_dir=str(tmp_path / "run"), device="cpu")
    cfg["model"].update(hidden_channels=32, filter_channels=64, inter_channels=32, upsample_initial_channel=32, n_layers=1)
    base = dict(batch_size=4, workers=0, max_steps=3, log_every=1, val_every=3, ckpt_every=100, gan_start_step=1)
    cfg["train"].update({**base, **train})
    return cfg

@pytest.mark.real_model
def test_multilingual_multispeaker_balanced_with_per_language_losses(tmp_path):
    """PLUMBING on synthetic data: language+speaker embeddings, language-balanced sampler, per-language loss reporting."""
    from ml.training.vits_trainer import Trainer
    _two_language_corpus(tmp_path / "d")
    t = Trainer(_tiny_cfg(tmp_path, language_balanced=True), resume=False)
    assert t.mc.n_languages == 2 and t.mc.n_speakers == 2 and t.G.g_ch > 0
    t.fit()
    log = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert any(set(e.get("lang_mel", {})) <= {"en", "xx"} and e.get("lang_mel") for e in log if e["event"] == "train")
    val = [e for e in log if e["event"] == "val"][0]
    assert set(val["val_mel_by_language"]) == {"en", "xx"} and all(np.isfinite(v) for v in val["val_mel_by_language"].values())

def test_language_balanced_sampler_equalises_languages():
    from ml.training.data import BucketSampler
    langs = ["en"] * 950 + ["hi"] * 50
    s = BucketSampler([100] * 1000, 10, languages=langs, balance=True)
    drawn = [langs[i] for b in s for i in b]
    assert 0.4 < drawn.count("hi") / len(drawn) < 0.6        # ~50/50 despite a 95/5 corpus
    plain = BucketSampler([100] * 1000, 10, languages=langs, balance=False)
    assert sorted(i for b in plain for i in b) == sorted(set(i for b in plain for i in b))   # no duplicates when unbalanced

@pytest.mark.real_model
def test_bf16_autocast_training_path_runs_and_is_finite(tmp_path):
    """bf16 autocast on CPU exercises the AMP code path. fp16/GradScaler and CUDA are NOT covered (no CUDA here)."""
    from ml.training.vits_trainer import Trainer
    _tiny_corpus(tmp_path / "d")
    t = Trainer(_tiny_cfg(tmp_path, amp="bf16"), resume=False); assert t.use_amp
    t.fit()
    log = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines() if '"train"' in l]
    assert len(log) == 3 and all(np.isfinite(e["mel"]) and np.isfinite(e["kl"]) and np.isfinite(e["g"]) for e in log)

def test_fp16_rejected_without_cuda(tmp_path):
    from ml.training.vits_trainer import Trainer
    _tiny_corpus(tmp_path / "d")
    with pytest.raises(ValueError, match="CUDA"): Trainer(_tiny_cfg(tmp_path, amp="fp16"), resume=False)

def test_char_durations_invariants():
    from ml.preprocessing.aligner import char_durations
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = int(rng.integers(3, 60)); n = int(rng.integers(L + 5, 400))
        start = np.sort(rng.choice(np.arange(n - 1), L, replace=False)); end = np.minimum(start + 1, n - 1)
        out_frames = int(rng.integers(2 * L + 3, 3 * n))
        for wb in (True, False):
            d = char_durations(start, end, n, wb, out_frames)
            assert d.sum() == out_frames and (d >= 1).all() and len(d) == (2 * L + 1 if wb else L)
        from ml.preprocessing.aligner import fs2_durations
        f = fs2_durations(start, end, n, out_frames); assert f.sum() == out_frames and (f >= 1).all() and len(f) == L + 2

@pytest.mark.real_model
def test_alignment_bootstrap_uses_external_durations_then_switches_to_mas(tmp_path, monkeypatch):
    from ml.training.vits_trainer import Trainer
    import ml.models.vits.model as vm
    _tiny_corpus(tmp_path / "d"); (tmp_path / "d" / "align").mkdir()
    for i in range(24):           # 11 chars ("hello there") aligned on a 100-frame CTC grid
        st = np.arange(11) * 8 + 5; np.savez(tmp_path / "d" / "align" / f"u{i}.npz", start=st, end=st + 4, n_frames=100, score=-1.0, text="HELLO|THERE")
    calls = []; real = vm.maximum_path
    monkeypatch.setattr(vm, "maximum_path", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    t = Trainer(_tiny_cfg(tmp_path, align_bootstrap_until=2, val_every=100), resume=False); t.fit()
    assert t.step == 3 and len(calls) == 1          # steps 0,1 used forced alignments; step 2 used MAS
    log = [json.loads(l) for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines() if '"train"' in l]
    assert [e["ext_used"] for e in log] == [1, 1, 0]
