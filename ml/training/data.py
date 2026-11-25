from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, Sampler
from ml.models.vits.audio_ops import Spectral
from ml.preprocessing.aligner import char_durations

class TTSDataset(Dataset):
    """Reads cached 16 kHz int16 waveforms; computes the linear spectrogram on CPU in the worker."""
    def __init__(self, root: Path, split: str, hop=256, n_fft=1024, sr=16000, min_frames=40, languages=None, speakers=None):
        self.root, self.hop = Path(root), hop
        self.rows = [r for r in map(json.loads, (self.root / "manifest.jsonl").read_text().splitlines()) if r["split"] == split]
        self.rows = [r for r in self.rows if int(r["duration_s"] * sr / hop) >= min_frames]
        self.spec = Spectral(sr=sr, n_fft=n_fft, hop=hop, win=n_fft)
        self.languages = languages or sorted({r.get("language", "en") for r in self.rows})
        self.speakers = speakers or sorted({r.get("speaker", "spk") for r in self.rows})
        self.frames = [int(r["duration_s"] * sr / hop) for r in self.rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        y = torch.from_numpy(np.load(self.root / "wavs16k" / f"{r['id']}.npy").astype(np.float32) / 32767.0)
        n = (len(y) // self.hop) * self.hop
        y = y[:n].view(1, 1, -1)
        with torch.no_grad(): spec = self.spec.linear(y)[0]
        ext = None
        ap = self.root / "align" / f"{r['id']}.npz"
        if ap.exists():
            a = np.load(ap)
            if 2 * len(a["start"]) + 1 == len(r["tokens"]):
                try: ext = torch.from_numpy(char_durations(a["start"], a["end"], int(a["n_frames"]), True, spec.size(1))).long()
                except (ValueError, AssertionError): ext = None
        return {"tokens": torch.tensor(r["tokens"]), "wav": y[0], "spec": spec, "id": r["id"], "ext_dur": ext,
                "lid": self.languages.index(r.get("language", "en")), "sid": self.speakers.index(r.get("speaker", "spk"))}

def collate(b):
    B = len(b)
    xl = torch.tensor([len(x["tokens"]) for x in b]); sl = torch.tensor([x["spec"].size(1) for x in b])
    tok = torch.zeros(B, int(xl.max()), dtype=torch.long)
    spec = torch.zeros(B, b[0]["spec"].size(0), int(sl.max())); wav = torch.zeros(B, 1, int(sl.max()) * 256)
    for i, x in enumerate(b):
        tok[i, :len(x["tokens"])] = x["tokens"]; spec[i, :, :x["spec"].size(1)] = x["spec"]; wav[i, :, :x["wav"].size(1)] = x["wav"]
    ext = None
    if all(x["ext_dur"] is not None for x in b):
        ext = torch.zeros(B, tok.size(1), dtype=torch.long)
        for i, x in enumerate(b): ext[i, :len(x["ext_dur"])] = x["ext_dur"]
    return {"ext_dur": ext, "tokens": tok, "tok_len": xl, "spec": spec, "spec_len": sl, "wav": wav,
            "lid": torch.tensor([x["lid"] for x in b]), "sid": torch.tensor([x["sid"] for x in b]), "ids": [x["id"] for x in b]}

class BucketSampler(Sampler):
    """Groups similar-length utterances to cut padding; reshuffles every epoch. Optional per-language balancing."""
    def __init__(self, frames, batch_size, seed=0, buckets=8, languages=None, balance=False):
        self.frames, self.bs, self.seed, self.epoch, self.nb = frames, batch_size, seed, 0, buckets
        self.languages, self.balance = languages, balance
    def set_epoch(self, e): self.epoch = e
    def _indices(self, rng):
        n = len(self.frames)
        if self.balance and self.languages:   # language-balanced: sample uniformly over languages, then within language
            by = {}
            for i, l in enumerate(self.languages): by.setdefault(l, []).append(i)
            return [rng.choice(by[rng.choice(list(by))]) for _ in range(n)]
        idx = list(range(n)); rng.shuffle(idx); return idx
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = self._indices(rng)
        chunk = self.bs * 50
        batches = []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=lambda i: self.frames[i])
            batches += [part[k:k + self.bs] for k in range(0, len(part), self.bs) if len(part[k:k + self.bs]) == self.bs]
        rng.shuffle(batches)
        return iter(batches)
    def __len__(self): return len(self.frames) // self.bs
