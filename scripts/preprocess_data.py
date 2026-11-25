#!/usr/bin/env python
"""Usage: preprocess_data.py ljspeech [--raw DIR] [--out DIR] [--archive ARCHIVE] [--workers N]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets import ljspeech

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["ljspeech"])
    ap.add_argument("--archive", default="data/datasets/ljspeech_raw/LJSpeech-1.1.tar.bz2")
    ap.add_argument("--raw", default="data/datasets/ljspeech_raw/extracted/LJSpeech-1.1")
    ap.add_argument("--out", default="data/datasets/ljspeech"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    raw = Path(a.raw)
    if not (raw / "metadata.csv").exists():
        print("extracting", a.archive, flush=True); ljspeech.extract(Path(a.archive), raw.parent)
    print(ljspeech.preprocess(raw, Path(a.out), workers=a.workers), flush=True)

if __name__ == "__main__":   # required: macOS multiprocessing uses 'spawn'
    main()
