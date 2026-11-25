#!/usr/bin/env python
"""Train FastSpeech 2 from scratch on teacher durations. Usage: train_fastspeech2.py --config configs/fs2_local.yaml [--no-resume] [--max-steps N]"""
import argparse, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from ml.training.fs2_trainer import FS2Trainer
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/fs2_local.yaml"); ap.add_argument("--no-resume", action="store_true"); ap.add_argument("--max-steps", type=int); ap.add_argument("--run-dir")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    if a.max_steps: cfg["train"]["max_steps"] = a.max_steps
    if a.run_dir: cfg["run_dir"] = a.run_dir
    FS2Trainer(cfg, resume=not a.no_resume).fit()

if __name__ == "__main__":
    main()
