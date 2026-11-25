#!/usr/bin/env python
"""Intelligibility + speed comparison on held-out LJSpeech test texts, via Whisper back-transcription.
Usage: evaluate.py --models mms-tts polyvoice-vits-lj [--n 30] [--out docs/evaluation_report]
Unmeasurable metrics are reported as null ("unavailable"), never estimated."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    from ml.inference import registry
    from ml.models.base import SynthesisRequest
    from ml.evaluation.backtranscribe import Transcriber
    from ml.evaluation.metrics import corpus, wer, cer
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--out", default="docs/evaluation_report")
    ap.add_argument("--whisper", default="openai/whisper-small"); a = ap.parse_args()
    root = Path("data/datasets/ljspeech")
    rows = [r for r in map(json.loads, (root / "manifest.jsonl").read_text().splitlines()) if r["split"] == "test"]
    rows = sorted(rows, key=lambda r: r["id"])[:a.n]           # deterministic subset of the held-out test split
    train_ref = next(r for r in map(json.loads, (root / "manifest.jsonl").read_text().splitlines()) if r["split"] == "train" and 4 < r["duration_s"] < 8)
    ref_audio = str(Path("data/eval/ref") / f"{train_ref['id']}.wav"); Path(ref_audio).parent.mkdir(parents=True, exist_ok=True)
    sf.write(ref_audio, np.load(root / "wavs16k" / f"{train_ref['id']}.npy").astype("float32") / 32767, 16000)
    asr = Transcriber(a.whisper); report = {"n_utterances": len(rows), "test_split": "LJSpeech held-out test (hash split)", "asr": a.whisper, "models": {}}
    refs = [r["text_norm"] for r in rows]
    for mid in a.models:
        b = registry.get_backend(mid); ok, why = b.is_available()
        if not ok: report["models"][mid] = {"skipped": why}; continue
        out_dir = Path("data/eval") / mid; out_dir.mkdir(parents=True, exist_ok=True)
        hyps, rtfs, durs, fails = [], [], [], 0
        for r in rows:
            try:
                req = SynthesisRequest(r["text_norm"], "en")
                if b.supported_features().get("requires_reference"): req.reference_audio, req.reference_text = ref_audio, train_ref["text_norm"]
                res = b.synthesize(req)
            except Exception as e:
                fails += 1; hyps.append(""); print(mid, r["id"], "FAILED", e, file=sys.stderr); continue
            sf.write(out_dir / f"{r['id']}.wav", res.audio, res.sample_rate)
            hyps.append(asr.transcribe(res.audio, res.sample_rate, "en")); rtfs.append(res.meta.get("rtf")); durs.append(len(res.audio) / res.sample_rate)
        b.release_model()
        rtfs = [x for x in rtfs if x is not None]
        report["models"][mid] = {"wer": corpus(refs, hyps, wer), "cer": corpus(refs, hyps, cer), "failures": fails,
            "mean_rtf": float(np.mean(rtfs)) if rtfs else None, "mean_audio_s": float(np.mean(durs)) if durs else None,
            "train_step": getattr(b, "step", None), "mos": None, "speaker_similarity": None, "first_audio_latency_s": None,
            "peak_memory_mb": None, "notes": "mos/speaker_similarity/latency/memory: unavailable (no listeners / embedder / streaming / per-process probe)",
            "examples": [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:3]]}
        print(mid, {k: report["models"][mid][k] for k in ("wer", "cer", "mean_rtf", "failures")}, flush=True)
    Path(a.out + ".json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    lines = ["# Evaluation report (measured)", f"{len(rows)} held-out LJSpeech test sentences; ASR = {a.whisper}. WER/CER lower is better. `-` = unavailable.", "",
             "| Model | WER | CER | mean RTF | failures | train step |", "|---|---|---|---|---|---|"]
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for m, v in report["models"].items():
        lines.append(f"| {m} | " + (f"skipped: {v['skipped']} | | | | |" if "skipped" in v else f"{f(v['wer'])} | {f(v['cer'])} | {f(v['mean_rtf'],2)} | {v['failures']} | {v['train_step'] or '-'} |"))
    Path(a.out + ".md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
