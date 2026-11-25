#!/usr/bin/env python
"""Indic intelligibility check using the per-language wav2vec2 CTC ASR models that were used as aligners (Whisper-small cannot read most Indic recordings).
Usage: eval_fs2_multi_ctc.py --step N [--langs hi bn ...] [--n 10] [--out docs/eval_fs2_multi_ctc.json]. CPU only. Reference = text_raw (native script); metric = CER/WER after
removing punctuation. The REAL recordings of the same sentences are scored by the same model as the ceiling. Models come from the local HF cache."""
import argparse, glob, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, librosa
torch.set_num_threads(3)
from ml.models.fastspeech2.model import FastSpeech2, FS2Config
from ml.preprocessing.tokenizer import CharTokenizer
from ml.evaluation.backtranscribe import _clean
from ml.evaluation.metrics import corpus, wer, cer
MODELS = {"hi": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200", "bn": "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice", "ta": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250",
          "te": "anuragshas/wav2vec2-large-xlsr-53-telugu", "pa": "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10", "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
          "mr": "sumedh/wav2vec2-large-xlsr-marathi", "as": "jayashreedevi2020/wav2vec2-large-xls-r-300m-assamese", "or": "theainerd/wav2vec2-large-xlsr-53-odia", "ml": "gvs/wav2vec2-large-xlsr-malayalam"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/fs2_multi_v1"); ap.add_argument("--step", type=int); ap.add_argument("--langs", nargs="+", default=list(MODELS))
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--out", default="docs/eval_fs2_multi_ctc.json"); ap.add_argument("--root", default="data/datasets/multi"); a = ap.parse_args()
    ck = (Path(a.run) / "ckpt" / f"step_{a.step:07d}.pt") if a.step else Path(sorted(glob.glob(f"{a.run}/ckpt/step_*.pt"))[-1])
    s = torch.load(ck, map_location="cpu", weights_only=False); net = FastSpeech2(FS2Config(**s["model_cfg"])); net.load_state_dict(s["net"]); net.eval()
    langs, smap, stats = s["languages"], s["speaker_map"], s["stats"]; tok = CharTokenizer.load(Path(a.root) / "tokenizer.json")
    from vocos import Vocos; from transformers import pipeline
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval(); res = {"checkpoint": str(ck), "step": s["step"], "languages": {}}
    for lang in a.langs:
        rows = [json.loads(l) for l in open(Path(a.root) / lang / "manifest.jsonl")]
        test = sorted([r for r in rows if r["split"] == "test" and r["speaker"] in smap], key=lambda r: r["id"])[: a.n]
        if not test: res["languages"][lang] = {"skipped": "no test rows"}; continue
        asr = pipeline("automatic-speech-recognition", model=MODELS[lang], device="cpu"); spk = [k for k, _ in Counter(r["speaker"] for r in rows if r["speaker"] in smap).most_common(2)]
        refs, hs, hr = [], [], []
        for i, r in enumerate(test):
            ids = tok.encode(r["text_norm"], lang, add_blank=False)
            if not ids: continue
            t = torch.tensor([[0] + ids + [0]])
            with torch.no_grad():
                o = net(t, torch.tensor([t.size(1)]), sid=torch.tensor([smap[spk[i % len(spk)]]]), lid=torch.tensor([langs.index(lang)]))
                mel = (o["mel"].float() * stats["mel_std"] + stats["mel_mean"]).transpose(1, 2); au = vocos.decode(mel)[0].numpy()
            real = np.load(Path(a.root) / lang / "wavs16k" / f"{r['id']}.npy").astype("float32") / 32767.0
            au16 = librosa.resample(au.astype("float32"), orig_sr=24000, target_sr=16000)
            refs.append(_clean(r["text_raw"])); hs.append(_clean(asr({"raw": au16, "sampling_rate": 16000})["text"])); hr.append(_clean(asr({"raw": real, "sampling_rate": 16000})["text"]))
        L = {"model": MODELS[lang], "n": len(refs), "synth": {"wer": corpus(refs, hs, wer), "cer": corpus(refs, hs, cer)}, "real": {"wer": corpus(refs, hr, wer), "cer": corpus(refs, hr, cer)},
             "examples": [{"ref": x, "synth": y, "real": z} for x, y, z in list(zip(refs, hs, hr))[:2]]}
        res["languages"][lang] = L; print(f"{lang}: synth WER {L['synth']['wer']:.3f} CER {L['synth']['cer']:.3f} | real WER {L['real']['wer']:.3f} CER {L['real']['cer']:.3f} (n={L['n']})", flush=True)
        del asr
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
