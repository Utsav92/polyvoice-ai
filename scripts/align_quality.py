#!/usr/bin/env python
"""Quality report for existing alignments: token-count agreement with the tokenizer, aligned speech span vs energy VAD (max(-35 dB, noise floor+10 dB)), CTC score. Usage: align_quality.py <lang> [--n 300]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, librosa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.preprocessing.tokenizer import CharTokenizer

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("lang"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--root", default="data/datasets/multi"); a = ap.parse_args()
    d = Path(a.root) / a.lang; tok = CharTokenizer.load(Path(a.root) / "tokenizer.json")
    rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]; rows = [r for r in rows if (d / "align" / f"{r['id']}.npz").exists()]
    rows = rows[:: max(len(rows) // a.n, 1)][: a.n]; mism = 0; lead = []; tail = []; ratio = []; sc = []
    for r in rows:
        al = np.load(d / "align" / f"{r['id']}.npz"); mism += int(len(tok.encode(r["text_norm"], a.lang, add_blank=False)) != len(al["start"]))
        y = np.load(d / "wavs16k" / f"{r['id']}.npy").astype(np.float32) / 32767
        e = librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0]; d_ = 20 * np.log10(e / (e.max() + 1e-9) + 1e-9); act = np.where(d_ > max(-35, np.percentile(d_, 10) + 10))[0]
        if not len(act): act = np.where(d_ > -35)[0]
        v0, v1 = act[0], act[-1] + 1; s0, s1 = al["start"][0], al["end"][-1] + 1
        lead.append((s0 - v0) * 0.02); tail.append((v1 - s1) * 0.02); ratio.append((s1 - s0) / max(v1 - v0, 1)); sc.append(float(al["score"]) / int(al["n_frames"]))
    print(json.dumps({"lang": a.lang, "n": len(rows), "token_count_mismatch": mism / len(rows), "lead_err_s_mean(aligned start - VAD start)": round(float(np.mean(lead)), 3),
        "tail_err_s_mean(VAD end - aligned end)": round(float(np.mean(tail)), 3), "abs_lead_s": round(float(np.mean(np.abs(lead))), 3), "abs_tail_s": round(float(np.mean(np.abs(tail))), 3),
        "speech_span_ratio_vs_vad": round(float(np.mean(ratio)), 3), "mean_logprob": round(float(np.mean(sc)), 3), "p05_logprob": round(float(np.percentile(sc, 5)), 3)}))

if __name__ == "__main__":
    main()
