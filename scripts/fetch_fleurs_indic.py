#!/usr/bin/env python
"""Fetch + preprocess FLEURS (google/fleurs, CC-BY-4.0, not gated) TRAIN split for Indian languages into data/datasets/multi/<lang>/.
Usage: fetch_fleurs_indic.py hi bn ta ... [--hours 7] [--min-free-gb 20]
Per language: download train.tar.gz + train.tsv, keep a deterministic hash-ordered subset up to --hours, validate with the same checks as LJSpeech,
write wavs16k/*.npy + manifest.jsonl (same schema as scripts/fetch_multilingual.py), then DELETE the archive. Test split is never touched (held out for eval).
FLEURS has no speaker id: 'speaker' is '<lang>_<gender>' (pseudo-speaker)."""
import argparse, csv, hashlib, io, json, shutil, sys, tarfile, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, soundfile as sf
from huggingface_hub import hf_hub_download
from ml.preprocessing.audio import load_validate, to_int16, AudioRejected
from ml.preprocessing.text_norm import normalize_text

CODE = {"hi": "hi_in", "bn": "bn_in", "ta": "ta_in", "te": "te_in", "mr": "mr_in", "gu": "gu_in", "kn": "kn_in", "ml": "ml_in", "pa": "pa_in", "or": "or_in", "as": "as_in", "ur": "ur_pk"}

def split_of(uid):
    h = int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 1000
    return "test" if h < 15 else "val" if h < 25 else "train"       # internal held-out slice of the TRAIN split (FLEURS' own test split stays untouched)

def run(lang, hours_cap, min_free, out_root, cache):
    fl = CODE[lang]; out = out_root / lang; wav = out / "wavs16k"; wav.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.jsonl"
    if (out / "fetch_report.json").exists(): print(lang, "already fetched (fetch_report.json present)", flush=True); return True
    if mf.exists() and sum(json.loads(l)["duration_s"] for l in mf.read_text().splitlines()) / 3600 >= hours_cap: print(lang, "already complete", flush=True); return
    if shutil.disk_usage(out).free / 2**30 < min_free: print("STOP low disk before", lang, flush=True); return False
    tsv = hf_hub_download("google/fleurs", f"data/{fl}/train.tsv", repo_type="dataset", cache_dir=cache)
    rows = [r for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 7 and r[-2].strip().isdigit()]   # robust to stray tabs/quotes inside text fields (pa_in)
    rows.sort(key=lambda r: hashlib.sha1(r[1].encode()).hexdigest())       # deterministic, speaker-agnostic order
    meta = {r[1]: r for r in rows}
    tgz = hf_hub_download("google/fleurs", f"data/{fl}/audio/train.tar.gz", repo_type="dataset", cache_dir=cache)
    want, hours, tot, rej = [], 0.0, 0, {}
    for r in rows:                                                          # budget by stated sample count (16 kHz)
        d = int(r[-2]) / 16000
        if hours + d / 3600 > hours_cap: continue
        want.append(r[1]); hours += d / 3600
    wantset = set(want); hours = 0.0; n = 0
    with tarfile.open(tgz, "r:gz") as tf, open(mf, "w", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
        for m in tf:
            name = Path(m.name).name
            if not m.isfile() or name not in wantset: continue
            data = tf.extractfile(m).read(); y, sr = sf.read(io.BytesIO(data), dtype="float32")
            tp = Path(tmp) / "x.wav"; sf.write(tp, y, sr)
            try: y16, rep = load_validate(str(tp), 16000)
            except AudioRejected as e: k = str(e).split(" (")[0]; rej[k] = rej.get(k, 0) + 1; continue
            r = meta[name]; text_raw = r[2]; tn = normalize_text(text_raw, lang)
            if len(tn) < 3: rej["empty text"] = rej.get("empty text", 0) + 1; continue
            uid = f"fleurs_{lang}_{name.replace('.wav','')}"
            np.save(wav / f"{uid}.npy", to_int16(y16)); hours += rep.duration_s / 3600; n += 1
            f.write(json.dumps({"id": uid, "language": lang, "speaker": f"{lang}_{r[-1].lower()}", "text_raw": text_raw, "text_norm": tn, "duration_s": rep.duration_s,
                                "peak": rep.peak, "rms_db": rep.rms_db, "clip_frac": rep.clip_frac, "silence_frac": rep.silence_frac,
                                "longest_silence_s": rep.longest_silence_s, "split": split_of(uid)}, ensure_ascii=False) + "\n")
    Path(tgz).resolve().unlink(missing_ok=True)                             # free the archive (cache blob)
    (out / "fetch_report.json").write_text(json.dumps({"language": lang, "source": "google/fleurs train", "hours": hours, "utts": n, "rejected": rej}, indent=1))
    print(json.dumps({"language": lang, "hours": round(hours, 2), "utts": n, "rejected": rej}), flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("langs", nargs="+", choices=list(CODE)); ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--out", default="data/datasets/multi"); ap.add_argument("--min-free-gb", type=float, default=20.0); a = ap.parse_args()
    cache = tempfile.mkdtemp(prefix="fleurs_cache_", dir=a.out)
    try:
        for l in a.langs:
            if run(l, a.hours, a.min_free_gb, Path(a.out), cache) is False: break
    finally: shutil.rmtree(cache, ignore_errors=True)

if __name__ == "__main__":
    main()
