#!/usr/bin/env python
"""Train the scaled-down VITS from random initialization.
Usage: train_vits.py --config configs/local_mps.yaml [--no-resume] [--max-steps N] [--run-dir DIR]"""
import argparse, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from ml.training.vits_trainer import Trainer
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/local_mps.yaml")
    ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--max-steps", type=int); ap.add_argument("--run-dir")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.max_steps: cfg["train"]["max_steps"] = a.max_steps
    if a.run_dir: cfg["run_dir"] = a.run_dir
    Trainer(cfg, resume=not a.no_resume).fit()

if __name__ == "__main__":   # required: macOS DataLoader workers use 'spawn' and re-import this file
    main()
