#!/usr/bin/env python
"""Stream AISHELL-3 (OpenSLR 93, Apache-2.0) and keep a deterministic ~1/KEEP subset of every speaker; the 19 GB archive is never stored.
Output: data/datasets/multi/zh/{wavs16k/*.npy, manifest.jsonl, fetch_report.json}. text_raw = Hanzi; text_norm = toned pinyin from the corpus
(syllables separated by spaces, tone digit 1-5 appended, e.g. 'guang3 zhou1'). Usage: fetch_aishell3.py [--keep 10] [--max-hours 11]"""
import argparse, hashlib, io, json, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ml.preprocessing.audio import load_validate, to_int16, AudioRejected
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_multilingual import split_of

URL = "https://www.openslr.org/resources/93/data_aishell3.tgz"

def parse_content(raw: bytes) -> dict:
    d = {}
    for line in raw.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3: continue
        toks = parts[1:]; hanzi = toks[0::2]; py = toks[1::2]
        if len(hanzi) != len(py): continue
        # punctuation entries ('%', '$') are prosody-break marks in the corpus; drop them
        pairs = [(h, p) for h, p in zip(hanzi, py) if re.fullmatch(r"[a-z]+[1-5]", p)]
        if pairs: d[parts[0].replace(".wav", "")] = ("".join(h for h, _ in pairs), " ".join(p for _, p in pairs))
    return d

def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/datasets/multi/zh"); ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--tmp", default="data/datasets/multi/_tmp_aishell3")
    ap.add_argument("--no-download", action="store_true", help="process whatever is already in --tmp (partial mirror download)")
    a = ap.parse_args(); out = Path(a.out); wav = out / "wavs16k"; wav.mkdir(parents=True, exist_ok=True)
    # OpenSLR itself served ~15 KB/s (19 GB impossible); the official 'AISHELL/AISHELL-3' HF mirror (apache-2.0, same files/layout) holds a ~3.4 GB subset.
    src = Path(a.tmp) if a.no_download else Path(snapshot_download("AISHELL/AISHELL-3", repo_type="dataset", local_dir=a.tmp, max_workers=4))
    content = {}
    for c in ("test/content.txt", "train/content.txt"): content.update(parse_content((src / c).read_bytes()))
    rej = {}; n = 0; hours = 0.0; spk = set(); nocontent = 0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for w in sorted(src.glob("*/wav/*/*.wav")):
            uid = w.stem
            if w.stat().st_size < 1000: continue          # partially downloaded file
            if uid not in content: nocontent += 1; continue
            if shutil.disk_usage(out).free / 2**30 < a.min_free_gb: print("STOP: low disk", flush=True); break
            try: y, rep = load_validate(str(w), 16000)
            except AudioRejected as e: k = str(e).split(" (")[0]; rej[k] = rej.get(k, 0) + 1; continue
            hz, py = content[uid]; sp = w.parent.name; uid2 = f"aishell3_{uid}"
            np.save(wav / f"{uid2}.npy", to_int16(y)); hours += rep.duration_s / 3600; n += 1; spk.add(sp)
            f.write(json.dumps({"id": uid2, "language": "zh", "speaker": f"zh_{sp}", "text_raw": hz, "text_norm": py, "duration_s": rep.duration_s, "peak": rep.peak,
                                "rms_db": rep.rms_db, "clip_frac": rep.clip_frac, "silence_frac": rep.silence_frac, "longest_silence_s": rep.longest_silence_s,
                                "split": split_of(uid2)}, ensure_ascii=False) + "\n")
            if n % 1000 == 0: print(f"zh: {n} utts {hours:.2f} h {len(spk)} spk", flush=True)
    rep = {"language": "zh", "hours": hours, "utts": n, "speakers": len(spk), "rejected": rej, "skipped_no_transcript": nocontent, "source": "HF AISHELL/AISHELL-3 subset"}
    (out / "fetch_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)
    shutil.rmtree(a.tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
