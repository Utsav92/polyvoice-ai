#!/usr/bin/env python
"""Forced-align LJSpeech characters with wav2vec2-base-960h (Apache-2.0). Writes data/datasets/ljspeech/align/{id}.npz + stats.json."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from ml.inference.device import detect_device
    from ml.preprocessing.aligner import ctc_align
    from ml.preprocessing.text_norm import normalize_text
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data/datasets/ljspeech"); ap.add_argument("--limit", type=int); ap.add_argument("--device")
    a = ap.parse_args(); root = Path(a.root); out = root / "align"; out.mkdir(exist_ok=True)
    dev = detect_device(a.device); name = "facebook/wav2vec2-base-960h"
    proc = Wav2Vec2Processor.from_pretrained(name); model = Wav2Vec2ForCTC.from_pretrained(name).eval().to(dev)
    vocab = proc.tokenizer.get_vocab()
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()][: a.limit]
    fails, scores, cdur, t0 = [], [], [], time.time()
    for i, r in enumerate(rows):
        if (out / f"{r['id']}.npz").exists(): continue
        txt = normalize_text(r["text_norm"], "en").upper().replace(" ", "|")
        try:
            tg = [vocab[c] for c in txt]
            y = np.load(root / "wavs16k" / f"{r['id']}.npy").astype(np.float32) / 32767
            x = torch.from_numpy((y - y.mean()) / (y.std() + 1e-7))[None].to(dev)
            with torch.no_grad(): lp = torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy()
            s, e, sc = ctc_align(lp, tg, blank=vocab["<pad>"])
        except Exception as ex:
            fails.append({"id": r["id"], "error": str(ex)[:120]}); continue
        np.savez(out / f"{r['id']}.npz", start=s, end=e, n_frames=lp.shape[0], score=sc, text=txt)
        scores.append(sc / lp.shape[0]); cdur.extend((np.diff(np.append(s, lp.shape[0])) * 20).tolist())
        if i % 1000 == 0: print(i, "/", len(rows), f"{time.time() - t0:.0f}s", flush=True)
    st = {"aligned": len(scores), "failed": len(fails), "model": name, "mean_logprob_per_frame": float(np.mean(scores)) if scores else None,
          "p05_logprob_per_frame": float(np.percentile(scores, 5)) if scores else None, "char_ms_median": float(np.median(cdur)) if cdur else None,
          "char_ms_p95": float(np.percentile(cdur, 95)) if cdur else None, "failures": fails[:20]}
    (out / "stats.json").write_text(json.dumps(st, indent=1)); print(st)

if __name__ == "__main__":
    main()
