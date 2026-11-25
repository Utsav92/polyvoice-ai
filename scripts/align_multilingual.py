#!/usr/bin/env python
"""Duration labels for the multilingual FastSpeech 2 data. CTC forced alignment with per-language wav2vec2 models that are all Apache-2.0:
en facebook/wav2vec2-base-960h, de/fr/it/es/nl/zh/ar jonatasgrosman/wav2vec2-large-xlsr-53-<language> (fine-tuned from XLSR-53 on Common Voice).
Labels only: the FS2 acoustic model is trained from scratch. Output per utterance: <lang>/align/<id>.npz with per-TOKEN start/end (same
format as scripts/align_dataset.py, so FS2Dataset works unchanged) + <lang>/align/stats.json.

Token <-> CTC-unit mapping is done with "groups": each group is (CTC units, number of FS2 tokens). Latin languages: 1 char = 1 group.
zh: the CTC model emits Hanzi, so a Hanzi owns the characters of its pinyin syllable. ar: the CTC model emits undiacritized Arabic letters, so a
word's letters own the characters of its uroman romanization (tokens are spread uniformly over the word span).
Usage: align_multilingual.py <lang> [--root data/datasets/multi] [--limit N] [--keep-model]"""
import argparse, json, re, shutil, sys, time, unicodedata
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = {l: f"jonatasgrosman/wav2vec2-large-xlsr-53-{n}" for l, n in
          dict(de="german", fr="french", it="italian", es="spanish", nl="dutch", zh="chinese-zh-cn", ar="arabic").items()}
MODELS["en"] = "facebook/wav2vec2-base-960h"
INDIC = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"}   # native-script CTC units -> uroman tokens (per-character split)
CENTERED = {"zh", "ar"}     # syllable/word-level CTC units: use spike centers for boundaries

def _unit(vocab, c):
    """CTC unit id for character c (space -> word delimiter), trying an accent-stripped fallback; None if the model has no such unit."""
    c = "|" if c == " " else c
    if c in vocab: return vocab[c]
    for cand in (c.upper(), unicodedata.normalize("NFD", c)[:1], unicodedata.normalize("NFD", c)[:1].upper()):
        if cand and cand in vocab: return vocab[cand]
    return None

def _vad_end(y, n_frames, db=-35.0):
    """Last 20 ms frame (CTC frame grid) whose RMS is within `db` dB of the loudest frame, +1."""
    import librosa
    r = librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0]; d_ = 20 * np.log10(r / (r.max() + 1e-9) + 1e-9); act = np.where(d_ > max(db, np.percentile(d_, 10) + 10))[0]   # adaptive: 10 dB above the noise floor (FLEURS is noisy)
    return float(min(act[-1] + 1, n_frames)) if len(act) else None

def _vad_span(y, n_frames, db=-35.0):
    import librosa
    r = librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0]; d_ = 20 * np.log10(r / (r.max() + 1e-9) + 1e-9); act = np.where(d_ > max(db, np.percentile(d_, 10) + 10))[0]   # adaptive: 10 dB above the noise floor (FLEURS is noisy)
    return (float(act[0]), float(min(act[-1] + 1, n_frames))) if len(act) else (0.0, float(n_frames))

def build_groups(lang, row, tok, vocab):
    """Returns (ids, units, groups): ids = FS2 token ids; units = CTC targets; groups = [(first_unit, n_units, n_tokens)]."""
    text = row["text_norm"]; prep = tok.prepare(text, lang)
    ids = [tok.idx[c] for c in prep if c in tok.idx]
    prep = "".join(c for c in prep if c in tok.idx)                       # the string that ids spell
    units, groups = [], []
    def add(us, nt): groups.append((len(units), len(us), nt)); units.extend(us)
    if lang == "zh":
        hanzi = [h for h in row["text_raw"] if not h.isspace()]; syl = prep.split(" ")
        if len(hanzi) != len(syl): raise ValueError(f"hanzi/syllable mismatch {len(hanzi)}/{len(syl)}")
        for k, (h, s) in enumerate(zip(hanzi, syl)):
            if k: add([], 1)                                               # the space between syllables
            u = vocab.get(h); add([u] if u is not None else [], len(s))
    elif lang == "ar":
        words = text.split(); rw = [tok.prepare(w, "ar") for w in words]
        rw = ["".join(c for c in w if c in tok.idx) for w in rw]
        if " ".join(w for w in rw if w) != prep: raise ValueError("word-wise romanization differs from sentence romanization")
        first = True
        for w, r in zip(words, rw):
            if not r: continue
            if not first: add([vocab["|"]], 1) if "|" in vocab else add([], 1)
            first = False
            base = "".join(ch for ch in unicodedata.normalize("NFKD", w) if not unicodedata.combining(ch))
            add([vocab[c] for c in base if c in vocab], len(r))
    elif lang in INDIC:
        clean = lambda t: "".join(c for c in tok.prepare(t, lang) if c in tok.idx); first = True
        for w in unicodedata.normalize("NFC", text).split():
            full = clean(w)
            if not full: continue
            if not first: add([vocab["|"]] if "|" in vocab else [], 1)
            first = False; prev = 0; pend = []
            for k, ch in enumerate(w, 1):
                cur = len(full) if k == len(w) else min(len(clean(w[:k])), len(full))
                dt = max(cur - prev, 0); prev = max(cur, prev); us = pend + ([vocab[ch]] if ch in vocab else [])
                if dt == 0: pend = us; continue         # e.g. virama / vowel sign: its time joins the next character that produces letters
                add(us, dt); pend = []
            if pend: f0, n0, t0 = groups[-1]; groups[-1] = (f0, n0 + len(pend), t0); units.extend(pend)
    else:
        for c in prep:
            u = _unit(vocab, c); add([u] if u is not None else [], 1)
    if sum(g[2] for g in groups) != len(ids): raise ValueError("group/token count mismatch")
    return ids, units, groups

def groups_to_token_spans(groups, start, end, n_frames, n_tok, centered=False, vad_end=None, tail_cap=None):
    """Unit spans -> per-token (start, end) in CTC frames. A group spans from its first unit to the next group's first unit (pauses are absorbed
    by the preceding group, like the English pipeline); unit-less groups get zero width (they receive 1 frame later); tokens split a span uniformly."""
    G = len(groups); gs = np.zeros(G); firsts = []
    pos = 0
    last = float(end[-1] + 1) if len(end) else float(n_frames)
    if tail_cap is not None and vad_end is not None and not centered: last = max(last, min(float(vad_end), last + tail_cap))   # CTC end = last spike; let the last token run on toward the energy end (<= tail_cap frames)
    nxt_unit_start = np.full(G + 1, last)
    for g in range(G - 1, -1, -1):
        f, n, _ = groups[g]
        nxt_unit_start[g] = float(start[f]) if n else nxt_unit_start[g + 1]
    if centered:
        # Peaky CTC emits one spike per Hanzi (a ~200 ms syllable), so spike start != syllable onset. Use spike centers; boundaries sit midway between neighbours.
        ug = [g for g in range(G) if groups[g][1]]
        if len(ug) >= 2:
            c = np.array([(start[groups[g][0]] + end[groups[g][0] + groups[g][1] - 1]) / 2.0 for g in ug])
            m = float(np.clip(np.median(np.diff(c)) / 2, 1, 12))                 # edge margin (frames), at most 240 ms
            gap = float(np.median(np.diff(c)))
            tail = m if vad_end is None else float(np.clip(vad_end - c[-1], m, max(gap, m)))   # CTC spikes sit early in a syllable: let energy extend the last one (<= 1 median gap)
            B = np.concatenate([[max(c[0] - m, 0)], (c[:-1] + c[1:]) / 2, [min(c[-1] + tail, n_frames)]])
            nxt_unit_start[G] = B[-1]
            for j, g in enumerate(ug): nxt_unit_start[g] = B[j]
            for g in range(G - 1, -1, -1):
                if not groups[g][1]: nxt_unit_start[g] = nxt_unit_start[g + 1]
    ts, te = np.zeros(n_tok), np.zeros(n_tok); k = 0
    for g, (f, n, nt) in enumerate(groups):
        b0 = nxt_unit_start[g]; b1 = nxt_unit_start[g + 1]
        for j in range(nt):
            ts[k] = b0 + (b1 - b0) * j / nt; te[k] = max(b0 + (b1 - b0) * (j + 1) / nt - 1e-3, ts[k]); k += 1
    return ts, te

def fetch_model(name):
    """Download only config/vocab/preprocessor json + ONE weight file (safetensors preferred); never the KenLM language_model/ files."""
    from huggingface_hub import snapshot_download, HfApi
    try:                                    # already cached? then never touch the network (the Hub API call can hang for minutes)
        p = snapshot_download(name, local_files_only=True)
        if (Path(p) / "vocab.json").exists() and ((Path(p) / "model.safetensors").exists() or (Path(p) / "pytorch_model.bin").exists()): return p
    except Exception: pass
    names = {x.rfilename for x in HfApi().model_info(name).siblings}
    w = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    return snapshot_download(name, allow_patterns=["*.json", w])

def main():
    import torch
    from transformers import Wav2Vec2ForCTC
    from huggingface_hub import snapshot_download
    from ml.inference.device import detect_device
    from ml.preprocessing.aligner import ctc_align
    from ml.preprocessing.tokenizer import CharTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("lang"); ap.add_argument("--root", default="data/datasets/multi"); ap.add_argument("--limit", type=int)
    ap.add_argument("--model", help="override the CTC model id (Indic candidates)"); ap.add_argument("--eval", type=int, help="evaluate on N evenly spread utterances, write nothing, print metrics")
    ap.add_argument("--license", default="Apache-2.0"); ap.add_argument("--align-dir", default="align", help="output dir name under the language dir (use align_v2 while training reads align/)"); ap.add_argument("--tail-extend", action="store_true", help="extend the last token toward the energy end (<= 200 ms); non-centered (Latin/Indic) languages"); ap.add_argument("--device"); ap.add_argument("--keep-model", action="store_true"); ap.add_argument("--tokenizer"); ap.add_argument("--split-shards", default="0/1")
    a = ap.parse_args(); root = Path(a.root); d = root / a.lang; out = d / a.align_dir; out.mkdir(exist_ok=True) if not a.eval else None
    tok = CharTokenizer.load(a.tokenizer or root / "tokenizer.json"); torch.set_num_threads(4); dev = torch.device(a.device or "cpu"); name = a.model or MODELS[a.lang]
    path = fetch_model(name)
    model = Wav2Vec2ForCTC.from_pretrained(path).eval().to(dev)
    vocab = json.loads((Path(path) / "vocab.json").read_text()); blank = model.config.pad_token_id if model.config.pad_token_id is not None else vocab.get("<pad>", 0)
    rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()][: a.limit]
    sh, nsh = map(int, a.split_shards.split("/")); rows = rows[sh::nsh]
    if a.eval: rows = rows[:: max(len(rows) // a.eval, 1)][: a.eval]; out = Path("/nonexistent")
    fails, scores, cdur, unmapped, t0, vad = [], [], [], [], time.time(), []
    for i, r in enumerate(rows):
        if (out / f"{r['id']}.npz").exists(): continue
        try:
            ids, units, groups = build_groups(a.lang, r, tok, vocab)
            y = np.load(d / "wavs16k" / f"{r['id']}.npy").astype(np.float32) / 32767
            x = torch.from_numpy((y - y.mean()) / (y.std() + 1e-7))[None].to(dev)
            with torch.no_grad(): lp = torch.log_softmax(model(x).logits[0].float(), -1).cpu().numpy()
            # ml.preprocessing.aligner.ctc_align assumes blank = column 0. HF xls-r fine-tunes have '|' at 0 and [PAD] last; fairseq/Vakyansh models have blank at 0
            # although config.pad_token_id says 1. The blank is the column that wins most frames, so find it from the emissions and move it to column 0.
            b = int(np.bincount(lp.argmax(1), minlength=lp.shape[1]).argmax())
            if b != 0:
                order = [b] + [c for c in range(lp.shape[1]) if c != b]; newid = {c: i for i, c in enumerate(order)}; lp = lp[:, order]; units = [newid[u] for u in units]
            s, e, sc = ctc_align(lp, units, blank=0)
            ts, te = groups_to_token_spans(groups, s, e, lp.shape[0], len(ids), centered=a.lang in CENTERED, vad_end=_vad_end(y, lp.shape[0]) if (a.lang in CENTERED or a.tail_extend) else None, tail_cap=10 if a.tail_extend else None)
        except Exception as ex:
            fails.append({"id": r["id"], "error": str(ex)[:120]}); continue
        v0, v1 = _vad_span(y, lp.shape[0]); vad.append(((ts[0] - v0) * 0.02, (v1 - (te[-1] + 1)) * 0.02, ((te[-1] + 1) - ts[0]) / max(v1 - v0, 1)))   # aligned vs energy VAD (s, s, ratio)
        if not a.eval: np.savez(out / f"{r['id']}.npz", start=ts, end=te, n_frames=lp.shape[0], score=sc, n_units=len(units), n_tokens=len(ids))
        scores.append(sc / lp.shape[0]); cdur.extend((np.diff(np.append(ts, lp.shape[0])) * 20).tolist()); unmapped.append(1 - len(units) / max(len(ids), 1))
        if i % 500 == 0: print(a.lang, i, "/", len(rows), f"{time.time() - t0:.0f}s fails={len(fails)}", flush=True)
    st = {"language": a.lang, "aligned": len(scores), "failed": len(fails), "model": name, "license": a.license, "vad_start_err_s_mean": float(np.mean([v[0] for v in vad])) if vad else None, "vad_end_err_s_mean": float(np.mean([v[1] for v in vad])) if vad else None,
          "vad_start_abs_err_s": float(np.mean([abs(v[0]) for v in vad])) if vad else None, "vad_end_abs_err_s": float(np.mean([abs(v[1]) for v in vad])) if vad else None,
          "speech_span_ratio_vs_vad": float(np.mean([v[2] for v in vad])) if vad else None, "fail_rate": len(fails) / max(len(fails) + len(scores), 1), "mean_logprob_per_frame": float(np.mean(scores)) if scores else None,
          "p05_logprob_per_frame": float(np.percentile(scores, 5)) if scores else None, "token_ms_median": float(np.median(cdur)) if cdur else None,
          "token_ms_p95": float(np.percentile(cdur, 95)) if cdur else None, "mean_frac_tokens_without_own_unit": float(np.mean(unmapped)) if unmapped else None,
          "wall_s": time.time() - t0, "failures": fails[:20]}
    if a.eval: print("EVAL", json.dumps(st)); return
    (out / ("stats.json" if nsh == 1 else f"stats_{sh}.json")).write_text(json.dumps(st, indent=1)); print(st)
    if not a.keep_model and a.lang != "en": shutil.rmtree(Path(path).parents[1], ignore_errors=True)       # free the ~1.2 GB cached model

if __name__ == "__main__":
    main()
