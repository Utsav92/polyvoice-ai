from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from ml.preprocessing.aligner import fs2_durations
from ml.preprocessing.tokenizer import CharTokenizer

class FS2Dataset(Dataset):
    """Token-level pitch/energy are the mean of frame values over each token's teacher-aligned frames (as in FastSpeech 2)."""
    def __init__(self, root: Path, split: str, stats: dict, languages=None, speakers=None, lang_dirs=None, tokenizer_path=None, min_score=None):
        """Single-corpus mode (default): root holds manifest.jsonl/align/fs2. Multilingual mode (opt-in): lang_dirs = per-language dirs, each with its own
        manifest.jsonl/align/fs2, plus one shared tokenizer_path; min_score drops utterances whose CTC alignment log-prob/frame is below it."""
        self.root = Path(root); self.stats = stats
        self.tok = CharTokenizer.load(tokenizer_path or self.root / "tokenizer.json")
        self.rows = []
        for d in ([Path(x) for x in lang_dirs] if lang_dirs else [self.root]):
            if not (d / "manifest.jsonl").exists(): continue                 # language declared but not downloaded yet: skipped, not an error
            for l in (d / "manifest.jsonl").read_text().splitlines():
                r = json.loads(l)
                if r["split"] != split or not (d / "align" / f"{r['id']}.npz").exists() or not (d / "fs2" / f"{r['id']}.npz").exists(): continue
                if min_score is not None:
                    a = np.load(d / "align" / f"{r['id']}.npz")
                    if float(a["score"]) / int(a["n_frames"]) < min_score: continue
                r["_dir"] = str(d); self.rows.append(r)
        if languages is not None: self.rows = [r for r in self.rows if r.get("language", "en") in languages]
        self.languages = languages or sorted({r.get("language", "en") for r in self.rows}); self.speakers = speakers or sorted({r.get("speaker", "spk") for r in self.rows})
        if speakers is not None: self.rows = [r for r in self.rows if r.get("speaker", "spk") in self.speakers]    # val speakers unseen in train cannot be embedded
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; d = Path(r.get("_dir", self.root)); f = np.load(d / "fs2" / f"{r['id']}.npz"); a = np.load(d / "align" / f"{r['id']}.npz")
        mel = f["mel"].astype(np.float32).T; T = mel.shape[0]
        dur = fs2_durations(a["start"], a["end"], int(a["n_frames"]), T)                 # [sil, chars..., sil], CTC-forced alignment
        ids = self.tok.encode(r["text_norm"], r.get("language", "en"), add_blank=False)
        tokens = [0] + ids + [0]; assert len(tokens) == len(dur), (r["id"], len(tokens), len(dur))
        f0, en = f["f0"][:T], f["energy"][:T]; bounds = np.concatenate([[0], np.cumsum(dur)])
        pitch = np.zeros(len(dur), np.float32); energy = np.zeros(len(dur), np.float32)
        for k in range(len(dur)):
            a, b = bounds[k], bounds[k + 1]
            if b <= a: continue
            v = f0[a:b][f0[a:b] > 0]
            if len(v): pitch[k] = (np.log(v).mean() - self.stats["lf0_mean"]) / self.stats["lf0_std"]
            energy[k] = (np.log(en[a:b] + 1e-5).mean() - self.stats["le_mean"]) / self.stats["le_std"]
        mel = (mel - self.stats["mel_mean"]) / self.stats["mel_std"]
        return {"tokens": torch.tensor(tokens), "dur": torch.from_numpy(dur).long(), "pitch": torch.from_numpy(pitch), "energy": torch.from_numpy(energy),
                "mel": torch.from_numpy(mel), "lid": self.languages.index(r.get("language", "en")), "sid": self.speakers.index(r.get("speaker", "spk")), "id": r["id"]}

def fs2_collate(b):
    B = len(b); xl = torch.tensor([len(x["tokens"]) for x in b]); ml = torch.tensor([x["mel"].size(0) for x in b])
    pad = lambda key, L: torch.stack([torch.nn.functional.pad(x[key], (0, L - len(x[key]))) for x in b])
    mel = torch.zeros(B, int(ml.max()), b[0]["mel"].size(1))
    for i, x in enumerate(b): mel[i, :x["mel"].size(0)] = x["mel"]
    L = int(xl.max())
    return {"tokens": pad("tokens", L), "tok_len": xl, "dur": pad("dur", L), "pitch": pad("pitch", L), "energy": pad("energy", L),
            "mel": mel, "mel_len": ml, "lid": torch.tensor([x["lid"] for x in b]), "sid": torch.tensor([x["sid"] for x in b]), "ids": [x["id"] for x in b]}

def compute_stats(root: Path, n=1500) -> dict:
    root = Path(root); rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines() if '"train"' in l][:n]
    m, s, mm, lf, le = 0.0, 0.0, 0, [], []
    mels = []
    for r in rows:
        f = np.load(root / "fs2" / f"{r['id']}.npz"); mels.append(f["mel"].astype(np.float32).ravel()[::7])
        v = f["f0"][f["f0"] > 0]; lf.append(np.log(v)); le.append(np.log(f["energy"] + 1e-5))
    mels, lf, le = np.concatenate(mels), np.concatenate(lf), np.concatenate(le)
    return {"mel_mean": float(mels.mean()), "mel_std": float(mels.std()), "lf0_mean": float(lf.mean()), "lf0_std": float(lf.std()),
            "le_mean": float(le.mean()), "le_std": float(le.std()), "n_utts": len(rows)}


def compute_stats_multi(lang_dirs, per_lang=400) -> dict:
    """Normalisation stats over ALL languages, equal weight per language (first `per_lang` train utterances with features each)."""
    mels, lf, le, n = [], [], [], 0
    for d in map(Path, lang_dirs):
        k = 0
        if not (d / "manifest.jsonl").exists(): continue
        for l in (d / "manifest.jsonl").read_text().splitlines():
            r = json.loads(l)
            if r["split"] != "train" or not (d / "fs2" / f"{r['id']}.npz").exists(): continue
            f = np.load(d / "fs2" / f"{r['id']}.npz"); mels.append(f["mel"].astype(np.float32).ravel()[::7])
            v = f["f0"][f["f0"] > 0]; lf.append(np.log(v)); le.append(np.log(f["energy"] + 1e-5)); k += 1
            if k >= per_lang: break
        n += k
    mels, lf, le = np.concatenate(mels), np.concatenate(lf), np.concatenate(le)
    return {"mel_mean": float(mels.mean()), "mel_std": float(mels.std()), "lf0_mean": float(lf.mean()), "lf0_std": float(lf.std()),
            "le_mean": float(le.mean()), "le_std": float(le.std()), "n_utts": n, "languages": [Path(d).name for d in lang_dirs]}


def language_coverage(d) -> dict:
    """Fraction of a language dir's manifest utterances that have BOTH alignment and features (all splits), plus hours/speakers of those."""
    d = Path(d); mf = d / "manifest.jsonl"
    if not mf.exists(): return {"utts": 0, "ready": 0, "coverage": 0.0, "hours": 0.0, "speakers": 0}
    n = k = 0; hrs = 0.0; spk = set()
    for l in mf.read_text().splitlines():
        r = json.loads(l); n += 1
        if (d / "align" / f"{r['id']}.npz").exists() and (d / "fs2" / f"{r['id']}.npz").exists(): k += 1; hrs += r["duration_s"] / 3600; spk.add(r.get("speaker"))
    return {"utts": n, "ready": k, "coverage": k / max(n, 1), "hours": round(hrs, 3), "speakers": len(spk)}
