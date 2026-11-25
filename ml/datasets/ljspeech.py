"""LJSpeech: safe extraction, metadata parsing, preprocessing to a reproducible manifest + cached 16 kHz int16 waveforms."""
from __future__ import annotations
import hashlib, json, tarfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from ml.preprocessing.audio import load_validate, to_int16, AudioRejected
from ml.preprocessing.tokenizer import english_tokenizer

def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as t:
        t.extractall(dest, filter="data")      # 'data' filter blocks absolute paths / traversal / special files

def split_of(utt_id: str, val=200, test=300, total=13100) -> str:
    """Deterministic hash split (no RNG state): ~val+test held-out utterances, so the held-out *texts* are unseen."""
    h = int(hashlib.sha1(utt_id.encode()).hexdigest(), 16) % total
    return "test" if h < test else "val" if h < test + val else "train"

def parse_metadata(root: Path):
    rows = []
    for line in (root / "metadata.csv").read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) >= 3: rows.append((parts[0], parts[1], parts[2]))
    return rows

def _work(args):
    utt, raw, norm, wav_path, out_dir, sr = args
    try:
        y, rep = load_validate(wav_path, sr)
    except AudioRejected as e:
        return {"id": utt, "rejected": str(e)}
    np.save(out_dir / f"{utt}.npy", to_int16(y))
    return {"id": utt, "text_raw": raw, "text_norm": norm, "duration_s": rep.duration_s, "peak": rep.peak,
            "rms_db": rep.rms_db, "clip_frac": rep.clip_frac, "silence_frac": rep.silence_frac,
            "longest_silence_s": rep.longest_silence_s, "split": split_of(utt)}

def preprocess(raw_root: Path, out_root: Path, sr=16000, hop=256, workers=6):
    wav_dir = out_root / "wavs16k"; wav_dir.mkdir(parents=True, exist_ok=True)
    tok = english_tokenizer(); tok.save(out_root / "tokenizer.json")
    jobs = [(u, r, n, raw_root / "wavs" / f"{u}.wav", wav_dir, sr) for u, r, n in parse_metadata(raw_root)]
    with Pool(workers) as p: res = p.map(_work, jobs, chunksize=64)
    kept, rejected = [], []
    for r in res:
        if "rejected" in r: rejected.append(r); continue
        ids = tok.encode(r["text_norm"], "en")
        frames = int(r["duration_s"] * sr / hop)
        if len(ids) == 0 or frames < len(ids): rejected.append({"id": r["id"], "rejected": f"tokens {len(ids)} > frames {frames}"}); continue
        r["tokens"] = ids; r["language"] = "en"; r["speaker"] = "lj"; kept.append(r)
    with open(out_root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in kept: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_root / "rejected.json").write_text(json.dumps(rejected, indent=1))
    stats = {"total": len(res), "kept": len(kept), "rejected": len(rejected),
             "hours_kept": sum(r["duration_s"] for r in kept) / 3600,
             "splits": {s: sum(1 for r in kept if r["split"] == s) for s in ("train", "val", "test")},
             "sample_rate": sr, "hop": hop, "vocab_size": tok.vocab_size}
    (out_root / "stats.json").write_text(json.dumps(stats, indent=1)); return stats
