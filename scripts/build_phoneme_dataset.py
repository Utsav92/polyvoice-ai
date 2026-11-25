#!/usr/bin/env python
"""Build a phoneme-token view of the LJSpeech dataset (reuses wavs16k via symlink; no forced alignments, MAS only).
Usage: build_phoneme_dataset.py [--src data/datasets/ljspeech] [--out data/datasets/ljspeech_ph]
Phonemes: espeak-ng en-us IPA (stress marks + punctuation kept), one symbol per Unicode character, blanks interspersed (VITS add_blank)."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.preprocessing.tokenizer import CharTokenizer

def phonemizer_fn():
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer import phonemize
    EspeakWrapper.set_library(espeakng_loader.get_library_path()); EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    return lambda texts: phonemize(texts, language="en-us", backend="espeak", strip=True, preserve_punctuation=True, with_stress=True, njobs=4)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data/datasets/ljspeech"); ap.add_argument("--out", default="data/datasets/ljspeech_ph")
    a = ap.parse_args(); src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (src / "manifest.jsonl").read_text().splitlines()]
    ph = phonemizer_fn()(["" + r["text_norm"] for r in rows])
    ph = [" ".join(p.split()) for p in ph]
    symbols = sorted({c for p in ph for c in p})
    tok = CharTokenizer(symbols); tok.save(out / "tokenizer.json")
    kept, rej = [], 0
    for r, p in zip(rows, ph):
        ids = [tok.idx[c] for c in p]; blanked = [0] * (2 * len(ids) + 1); blanked[1::2] = ids
        if int(r["duration_s"] * 16000 / 256) < len(blanked): rej += 1; continue
        r = dict(r); r["phonemes"] = p; r["tokens"] = blanked; kept.append(r)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    if not (out / "wavs16k").exists(): os.symlink((src / "wavs16k").resolve(), out / "wavs16k")
    print(json.dumps({"utterances": len(kept), "rejected_too_long": rej, "vocab": tok.vocab_size, "symbols": "".join(symbols), "mean_tokens": sum(len(r["tokens"]) for r in kept) / len(kept)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
