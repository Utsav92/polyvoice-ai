#!/usr/bin/env python
"""Compute Vocos-compatible mel + F0 + energy for FastSpeech 2. Usage: extract_fs2_features.py [--workers N]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from ml.preprocessing import fs2_features
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--multi-lang-dir", nargs="+", help="multilingual mode: language dirs (e.g. data/datasets/multi/de) with wavs16k/ + manifest.jsonl")
    a = ap.parse_args()
    if a.multi_lang_dir:
        for d in a.multi_lang_dir: print(fs2_features.build_from_cache(Path(d), a.workers), flush=True)
        return
    print(fs2_features.build(Path("data/datasets/ljspeech_raw/extracted/LJSpeech-1.1/wavs"), Path("data/datasets/ljspeech/manifest.jsonl"),
                             Path("data/datasets/ljspeech/fs2"), a.workers), flush=True)

if __name__ == "__main__":   # macOS multiprocessing 'spawn' re-imports this file
    main()
