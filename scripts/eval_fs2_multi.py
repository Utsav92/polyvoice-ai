#!/usr/bin/env python
"""Per-language intelligibility check of a multilingual FS2 checkpoint (CPU only, safe to run next to training).
Usage: eval_fs2_multi.py [--run runs/fs2_multi_v1] [--step N] [--langs en de zh] [--n 12] [--out docs/eval_fs2_multi.json]
For each language: n held-out test sentences (manifest split == 'test'), synthesised with that language's id and up to 3 of its speakers (language-balanced),
decoded with Vocos, back-transcribed with Whisper-small; the SAME sentences' real recordings are transcribed too as the ceiling.
Reference text = text_raw (zh: hanzi; metrics compare characters). Nothing is estimated: failures are reported as such."""
import argparse, glob, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
torch.set_num_threads(3)
from ml.models.fastspeech2.model import FastSpeech2, FS2Config
from ml.preprocessing.tokenizer import CharTokenizer
from ml.evaluation.backtranscribe import Transcriber
from ml.evaluation.metrics import corpus, wer, cer

def zh_pinyin(text):
    """Script-independent comparison for Chinese: toned pinyin syllables (simplified/traditional and hanzi variants collapse; digits/Latin dropped)."""
    from pypinyin import lazy_pinyin, Style
    return " ".join(lazy_pinyin("".join(c for c in text if '\u4e00' <= c <= '\u9fff'), style=Style.TONE3))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/fs2_multi_v1"); ap.add_argument("--step", type=int); ap.add_argument("--langs", nargs="+", default=["en", "de", "zh"])
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="docs/eval_fs2_multi.json"); ap.add_argument("--root", default="data/datasets/multi"); a = ap.parse_args()
    ck = (Path(a.run) / "ckpt" / f"step_{a.step:07d}.pt") if a.step else Path(sorted(glob.glob(f"{a.run}/ckpt/step_*.pt"))[-1])
    s = torch.load(ck, map_location="cpu", weights_only=False)
    net = FastSpeech2(FS2Config(**s["model_cfg"])); net.load_state_dict(s["net"]); net.eval()
    langs, smap, stats = s["languages"], s["speaker_map"], s["stats"]; tok = CharTokenizer.load(Path(a.root) / "tokenizer.json")
    from vocos import Vocos
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval(); asr = Transcriber("openai/whisper-small")
    res = {"checkpoint": str(ck), "step": s["step"], "languages": {}}
    for lang in a.langs:
        rows = [json.loads(l) for l in open(Path(a.root) / lang / "manifest.jsonl")]
        test = sorted([r for r in rows if r["split"] == "test" and r["speaker"] in smap], key=lambda r: r["id"])[: a.n * 3]
        if not test or lang not in langs: res["languages"][lang] = {"skipped": "no test rows with a known speaker, or language not in checkpoint"}; continue
        spk = [k for k, _ in Counter(r["speaker"] for r in rows if r["speaker"] in smap).most_common(3)]
        sel = test[: a.n]; refs, hs, hr = [], [], []
        for i, r in enumerate(sel):
            sp = spk[i % len(spk)]
            ids = tok.encode(r["text_norm"], lang, add_blank=False)
            if not ids: continue
            tokens = torch.tensor([[0] + ids + [0]]); tl = torch.tensor([tokens.size(1)])
            with torch.no_grad():
                o = net(tokens, tl, sid=torch.tensor([smap[sp]]), lid=torch.tensor([langs.index(lang)]))
                mel = (o["mel"].float() * stats["mel_std"] + stats["mel_mean"]).transpose(1, 2); audio = vocos.decode(mel)[0].numpy()
            real = np.load(Path(a.root) / lang / "wavs16k" / f"{r['id']}.npy").astype("float32") / 32767.0
            refs.append(r["text_raw"]); hs.append(asr.transcribe(audio, 24000, lang)); hr.append(asr.transcribe(real, 16000, lang))
        if lang == "zh": refs, hs, hr = [zh_pinyin(x) for x in refs], [zh_pinyin(x) for x in hs], [zh_pinyin(x) for x in hr]; hs0 = hs   # score on pinyin
        res["languages"][lang] = {"n": len(refs), "speakers": spk, "synth": {"wer": corpus(refs, hs, wer), "cer": corpus(refs, hs, cer)}, "real": {"wer": corpus(refs, hr, wer), "cer": corpus(refs, hr, cer)},
                                  "examples": [{"ref": r_, "synth": h_.strip(), "real": g_.strip()} for r_, h_, g_ in list(zip(refs, hs, hr))[:3]]}
        L = res["languages"][lang]; print(f"{lang}: synth WER {L['synth']['wer']:.3f} CER {L['synth']['cer']:.3f} | real WER {L['real']['wer']:.3f} CER {L['real']['cer']:.3f} (n={L['n']})", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()
