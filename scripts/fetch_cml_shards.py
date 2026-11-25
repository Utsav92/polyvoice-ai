#!/usr/bin/env python
"""Shard-based CML-TTS fetch (ylacombe/cml-tts, CC-BY-4.0): robust alternative to HF streaming (which stalled on dropped connections).
Usage: HF_HUB_DISABLE_XET=1 fetch_cml_shards.py german french spanish italian dutch [--hours 10] [--shards 6] [--min-free-gb 15]
Per language: pick --shards evenly spaced train parquet shards (~345 MB / ~2 h each, spreads speakers), download one at a time, keep a seeded shuffle of
quality-filtered utterances (levenshtein >= 0.9) up to hours/shards per shard, validate like LJSpeech, write data/datasets/multi/<lang>/{wavs16k,manifest.jsonl}, DELETE the shard."""
import argparse, hashlib, io, json, re, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, soundfile as sf, pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from ml.preprocessing.audio import load_validate, to_int16, AudioRejected
from ml.preprocessing.text_norm import normalize_text

CODE = {"german": "de", "french": "fr", "italian": "it", "spanish": "es", "dutch": "nl"}
def split_of(uid):
    h = int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 1000
    return "test" if h < 15 else "val" if h < 25 else "train"

def shard_list(lang):
    out = []
    for s in HfApi().dataset_info("ylacombe/cml-tts", files_metadata=True).siblings:
        m = re.match(rf"{lang}/train-(\d+)-of-\d+-\w+\.parquet", s.rfilename)
        if m: out.append((int(m.group(1)), s.rfilename))
    return [f for _, f in sorted(out)]

def run(lang, hours_cap, n_shards, min_free, out_root, cache):
    code = CODE[lang]; out = out_root / code; wav = out / "wavs16k"; wav.mkdir(parents=True, exist_ok=True); mf = out / "manifest.jsonl"
    rows_done = [json.loads(l) for l in mf.read_text().splitlines()] if mf.exists() else []
    seen = {r["id"] for r in rows_done}; hours = sum(r["duration_s"] for r in rows_done) / 3600
    if hours >= hours_cap: print(code, "already complete", flush=True); return True
    shards = shard_list(lang); picks = [shards[int(i)] for i in np.linspace(len(shards) * 0.05, len(shards) * 0.95, n_shards)]
    per = hours_cap / n_shards; rej = {}; spk = set(r["speaker"] for r in rows_done); n = 0
    for k, fn in enumerate(picks):
        if hours >= hours_cap: break
        if shutil.disk_usage(out).free / 2**30 < min_free: print("STOP low disk", flush=True); return False
        shard_target = min(hours_cap, (k + 1) * per)
        for attempt in range(6):
            try: p = hf_hub_download("ylacombe/cml-tts", fn, repo_type="dataset", cache_dir=cache); break
            except Exception as e: print(code, "download retry", attempt, type(e).__name__, str(e)[:80], flush=True)
        else: print(code, "giving up on shard", fn, flush=True); continue
        t = pq.read_table(p, columns=["audio", "text", "levenshtein", "speaker_id", "duration"]).to_pylist()
        rng = np.random.default_rng(k); rng.shuffle(t)
        with open(mf, "a", encoding="utf-8") as f, tempfile.TemporaryDirectory() as tmp:
            for x in t:
                if hours >= shard_target: break
                if float(x["levenshtein"]) < 0.9: continue
                name = (x["audio"].get("path") or "").replace(".wav", "")
                uid = f"cml_{code}_{name}"
                if not name or uid in seen: continue
                y, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32"); tp = Path(tmp) / "x.wav"; sf.write(tp, y, sr)
                try: y16, rep = load_validate(str(tp), 16000)
                except AudioRejected as e: kk = str(e).split(" (")[0]; rej[kk] = rej.get(kk, 0) + 1; continue
                tn = normalize_text(x["text"], code)
                if len(tn) < 3: rej["empty text"] = rej.get("empty text", 0) + 1; continue
                np.save(wav / f"{uid}.npy", to_int16(y16)); seen.add(uid); hours += rep.duration_s / 3600; n += 1; spk.add(f"{code}_{x['speaker_id']}")
                f.write(json.dumps({"id": uid, "language": code, "speaker": f"{code}_{x['speaker_id']}", "text_raw": x["text"], "text_norm": tn, "duration_s": rep.duration_s,
                                    "peak": rep.peak, "rms_db": rep.rms_db, "clip_frac": rep.clip_frac, "silence_frac": rep.silence_frac,
                                    "longest_silence_s": rep.longest_silence_s, "split": split_of(uid)}, ensure_ascii=False) + "\n"); f.flush()
        del t; Path(p).resolve().unlink(missing_ok=True)
        print(f"{code}: shard {k+1}/{n_shards} done, {hours:.2f} h, {len(spk)} speakers", flush=True)
    (out / "fetch_report.json").write_text(json.dumps({"language": code, "source": "ylacombe/cml-tts (shards)", "hours": hours, "speakers": len(spk), "rejected": rej}, indent=1))
    print(json.dumps({"language": code, "hours": round(hours, 2), "new_utts": n, "speakers": len(spk), "rejected": rej}), flush=True); return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("langs", nargs="+", choices=list(CODE)); ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--shards", type=int, default=6); ap.add_argument("--out", default="data/datasets/multi"); ap.add_argument("--min-free-gb", type=float, default=15.0)
    a = ap.parse_args(); cache = tempfile.mkdtemp(prefix="cml_cache_", dir=a.out)
    try:
        for l in a.langs:
            if run(l, a.hours, a.shards, a.min_free_gb, Path(a.out), cache) is False: break
    finally: shutil.rmtree(cache, ignore_errors=True)

if __name__ == "__main__":
    main()
