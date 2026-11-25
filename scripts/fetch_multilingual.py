#!/usr/bin/env python
"""Stream-and-preprocess approved multilingual TTS data into data/datasets/multi/<lang>/ (wavs16k/*.npy + manifest.jsonl).
Usage: fetch_multilingual.py cml <german|french|italian|spanish|dutch> [--hours 10] | clartts [--hours 12]
Only datasets approved in configs/datasets_manifest.json. Streams from Hugging Face (no full archive download), shuffles across
speakers, filters by transcript quality + the same audio validation as LJSpeech, caps total hours, and aborts if free disk < --min-free-gb."""
import argparse, hashlib, io, json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, soundfile as sf, soxr
from ml.preprocessing.audio import load_validate, to_int16, AudioRejected
from ml.preprocessing.text_norm import normalize_text

LANG = {"german": "de", "french": "fr", "italian": "it", "spanish": "es", "dutch": "nl"}

def split_of(uid: str) -> str:
    h = int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 1000
    return "test" if h < 15 else "val" if h < 25 else "train"

def iter_source(kind, lang_name):
    from datasets import load_dataset, Audio
    if kind == "cml":
        ds = load_dataset("ylacombe/cml-tts", lang_name, split="train", streaming=True).cast_column("audio", Audio(decode=False))
        ds = ds.shuffle(seed=0, buffer_size=3000)
        for x in ds:
            if float(x["levenshtein"]) < 0.9: continue                       # transcript agrees with wav2vec2 ASR
            y, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32")
            yield f"cml_{LANG[lang_name]}_{x['audio']['path'].replace('.wav','')}", y, sr, x["text"], str(x["speaker_id"])
    else:
        ds = load_dataset("MBZUAI/ClArTTS", split="train", streaming=True)
        for x in ds:
            y = np.asarray(x["audio"], dtype="float32"); sr = int(float(x["sampling_rate"]))
            if abs(len(y) / sr - float(x["duration"])) > 0.1: continue        # sampling-rate field must agree with stated duration
            yield f"clartts_{x['file'].replace('.wav','')}", y, sr, x["text"], "ar_m1"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["cml", "clartts"]); ap.add_argument("lang", nargs="?")
    ap.add_argument("--hours", type=float, default=10.0); ap.add_argument("--out", default="data/datasets/multi"); ap.add_argument("--min-free-gb", type=float, default=20.0)
    a = ap.parse_args(); code = "ar" if a.kind == "clartts" else LANG[a.lang]
    out = Path(a.out) / code; wav = out / "wavs16k"; wav.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.jsonl"; seen = {json.loads(l)["id"] for l in mf.read_text().splitlines()} if mf.exists() else set()
    hours = sum(json.loads(l)["duration_s"] for l in mf.read_text().splitlines()) / 3600 if mf.exists() else 0.0
    rej = {}; n = 0; spk = set()
    with open(mf, "a", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
        for uid, y, sr, text, speaker in iter_source(a.kind, a.lang):
            if hours >= a.hours: break
            if uid in seen: continue
            if shutil.disk_usage(out).free / 2**30 < a.min_free_gb: print("STOP: low disk", flush=True); break
            tp = Path(tmp) / "x.wav"; sf.write(tp, y, sr)
            try: y16, rep = load_validate(str(tp), 16000)
            except AudioRejected as e: k = str(e).split(" (")[0]; rej[k] = rej.get(k, 0) + 1; continue
            tn = normalize_text(text, code)
            if len(tn) < 3: rej["empty text"] = rej.get("empty text", 0) + 1; continue
            np.save(wav / f"{uid}.npy", to_int16(y16)); hours += rep.duration_s / 3600; n += 1; spk.add(speaker)
            f.write(json.dumps({"id": uid, "language": code, "speaker": f"{code}_{speaker}", "text_raw": text, "text_norm": tn, "duration_s": rep.duration_s,
                                "peak": rep.peak, "rms_db": rep.rms_db, "clip_frac": rep.clip_frac, "silence_frac": rep.silence_frac,
                                "longest_silence_s": rep.longest_silence_s, "split": split_of(uid)}, ensure_ascii=False) + "\n"); f.flush()
            if n % 200 == 0: print(f"{code}: {n} utts {hours:.2f} h {len(spk)} speakers", flush=True)
    (out / "fetch_report.json").write_text(json.dumps({"language": code, "hours": hours, "new_utts": n, "speakers_in_run": len(spk), "rejected": rej}, indent=1))
    print(json.dumps({"language": code, "hours": round(hours, 2), "new_utts": n, "speakers": len(spk), "rejected": rej}), flush=True)

if __name__ == "__main__":
    main()
