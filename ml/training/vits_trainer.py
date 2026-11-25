from __future__ import annotations
import json, time, glob, os
from dataclasses import asdict
from pathlib import Path
import torch, soundfile as sf, yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ml.inference.device import detect_device, memory_allocated_mb
from ml.models.vits.model import VITS, VitsConfig, feature_loss, discriminator_loss, generator_loss
from ml.models.vits.modules import MultiPeriodDiscriminator
from ml.models.vits.audio_ops import Spectral
from ml.models.vits.commons import slice_segments
from ml.preprocessing.tokenizer import CharTokenizer
from ml.training.data import TTSDataset, collate, BucketSampler

def _tupleize(d):
    return {k: (tuple(tuple(i) if isinstance(i, list) else i for i in v) if isinstance(v, list) else v) for k, v in d.items()}

class Trainer:
    def __init__(self, cfg: dict, resume: bool = True):
        self.cfg, t = cfg, cfg["train"]
        self.dev = detect_device(cfg.get("device"))
        self.run = Path(cfg["run_dir"]); (self.run / "ckpt").mkdir(parents=True, exist_ok=True); (self.run / "samples").mkdir(exist_ok=True)
        root = Path(cfg["data_root"])
        self.tok = CharTokenizer.load(root / "tokenizer.json")
        self.tr = TTSDataset(root, "train"); self.va = TTSDataset(root, "val", languages=self.tr.languages, speakers=self.tr.speakers)
        mc = VitsConfig(**_tupleize(cfg["model"]), n_vocab=self.tok.vocab_size, n_speakers=len(self.tr.speakers), n_languages=len(self.tr.languages))
        self.mc = mc
        self.G = VITS(mc).to(self.dev); self.D = MultiPeriodDiscriminator().to(self.dev); self.spec = Spectral(mc.sample_rate, mc.n_fft, mc.hop, mc.n_fft, mc.n_mels).to(self.dev)
        betas = tuple(t.get("betas", (0.8, 0.99)))
        self.oG = torch.optim.AdamW(self.G.parameters(), t["lr"], betas=betas, eps=1e-9)
        self.oD = torch.optim.AdamW(self.D.parameters(), t["lr"], betas=betas, eps=1e-9)
        self.sG = torch.optim.lr_scheduler.ExponentialLR(self.oG, t["lr_decay"]); self.sD = torch.optim.lr_scheduler.ExponentialLR(self.oD, t["lr_decay"])
        amp = t.get("amp", "none")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        if amp == "fp16" and self.dev.type != "cuda": raise ValueError("fp16 autocast requires CUDA; use bf16 or none")
        self.use_amp = self.amp_dtype is not None and self.dev.type in ("cuda", "cpu")
        if amp != "none" and not self.use_amp: print(f"[warn] amp={amp} ignored on {self.dev.type}; training in fp32")
        self.scG = torch.amp.GradScaler("cuda", enabled=(amp == "fp16")); self.scD = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))
        self.step, self.epoch = 0, 0
        if resume: self.load_latest()
        self.log_f = open(self.run / "train_log.jsonl", "a")
        self.log(event="resume" if self.step else "start", step=self.step, batch_size=t["batch_size"], align_bootstrap_until=t.get("align_bootstrap_until", 0))
        (self.run / "config.yaml").write_text(yaml.safe_dump(cfg))

    # ---- checkpointing
    def save(self):
        p = self.run / "ckpt" / f"step_{self.step:07d}.pt"
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict(), "oG": self.oG.state_dict(), "oD": self.oD.state_dict(),
                    "sG": self.sG.state_dict(), "sD": self.sD.state_dict(), "step": self.step, "epoch": self.epoch,
                    "model_cfg": asdict(self.mc), "languages": self.tr.languages, "speakers": self.tr.speakers}, p)
        for old in sorted(glob.glob(str(self.run / "ckpt" / "step_*.pt")))[:-self.cfg["train"].get("keep_ckpts", 3)]: os.remove(old)
        return p
    def load_latest(self):
        cks = sorted(glob.glob(str(self.run / "ckpt" / "step_*.pt")))
        if not cks: return
        s = torch.load(cks[-1], map_location="cpu", weights_only=False)
        self.G.load_state_dict(s["G"]); self.D.load_state_dict(s["D"]); self.oG.load_state_dict(s["oG"]); self.oD.load_state_dict(s["oD"])
        self.sG.load_state_dict(s["sG"]); self.sD.load_state_dict(s["sD"]); self.step, self.epoch = s["step"], s["epoch"]
        print(f"resumed from {cks[-1]} at step {self.step}")

    def log(self, **kw):
        kw["time"] = time.time(); self.log_f.write(json.dumps(kw) + "\n"); self.log_f.flush()

    def _dev(self, b): return {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in b.items()}

    # ---- one micro-batch; returns dict of float losses
    def micro(self, b, accum, gan: bool):
        c = self.mc; t = self.cfg["train"]
        ext = b.get("ext_dur")
        if ext is not None and self.step < t.get("align_bootstrap_until", 0): ext = ext.to(self.dev)
        else: ext = None
        b = self._dev({k: v for k, v in b.items() if k != "ext_dur"})
        ac = lambda: torch.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.use_amp)
        with ac():
            o, l_dur, l_kl, attn, starts, _ = self.G(b["tokens"], b["tok_len"], b["spec"], b["spec_len"], b["sid"], b["lid"], ext_dur=ext)
        o, l_dur, l_kl = o.float(), l_dur.float(), l_kl.float()
        y = slice_segments(b["wav"], starts * c.hop, c.segment_size)
        y_mel, yh_mel = self.spec.mel(y), self.spec.mel(o)
        per = (yh_mel - y_mel).abs().mean((1, 2)).detach()
        self.last_lang_mel = {self.tr.languages[int(l)]: float(per[b["lid"] == l].mean()) for l in b["lid"].unique()}
        l_mel = F.l1_loss(yh_mel, y_mel) * t["c_mel"]
        out = {"ext_used": int(ext is not None)}
        if gan:
            self.D.requires_grad_(True)
            with ac(): r, g, _, _ = self.D(y, o.detach())
            l_d = discriminator_loss([x.float() for x in r], [x.float() for x in g])
            self.scD.scale(l_d / accum).backward(); out["d"] = float(l_d)
            self.D.requires_grad_(False)
            with ac(): r, g, fr, fg = self.D(y, o)
            g = [x.float() for x in g]; fr = [[a.float() for a in d] for d in fr]; fg = [[a.float() for a in d] for d in fg]
            l_fm, l_adv = feature_loss(fr, fg), generator_loss(g)
            l_g = l_mel + l_kl * t["c_kl"] + l_dur + l_fm + l_adv; out.update(fm=float(l_fm), adv=float(l_adv))
            self.D.requires_grad_(True)
        else:
            l_g = l_mel + l_kl * t["c_kl"] + l_dur
        self.scG.scale(l_g / accum).backward()
        out.update(self.G.last_diag)
        out.update(g=float(l_g), mel=float(l_mel / t["c_mel"]), kl=float(l_kl), dur=float(l_dur))
        return out

    @torch.no_grad()
    def validate(self, n=64):
        self.G.eval(); tot = {"mel": 0.0, "kl": 0.0, "dur": 0.0}; k = 0; lang_sum, lang_n = {}, {}
        dl = DataLoader(self.va, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)
        c = self.mc
        for i, b in enumerate(dl):
            if i * 8 >= n: break
            b = self._dev(b)
            o, l_dur, l_kl, _, starts, _ = self.G(b["tokens"], b["tok_len"], b["spec"], b["spec_len"], b["sid"], b["lid"])
            y = slice_segments(b["wav"], starts * c.hop, c.segment_size)
            per = (self.spec.mel(o) - self.spec.mel(y)).abs().mean((1, 2))
            for l in b["lid"].unique():
                nm = self.tr.languages[int(l)]; m = b["lid"] == l
                lang_sum[nm] = lang_sum.get(nm, 0.0) + float(per[m].sum()); lang_n[nm] = lang_n.get(nm, 0) + int(m.sum())
            tot["mel"] += float(per.mean()); tot["kl"] += float(l_kl); tot["dur"] += float(l_dur); k += 1
        self.G.train(); r = {f"val_{a}": v / max(k, 1) for a, v in tot.items()}
        r["val_mel_by_language"] = {l: lang_sum[l] / lang_n[l] for l in lang_sum}; return r

    @torch.no_grad()
    def synth_samples(self):
        self.G.eval()
        for i, text in enumerate(self.cfg.get("sample_texts", ["The quick brown fox jumps over the lazy dog."])):
            ids = torch.tensor([self.tok.encode(text, "en")], device=self.dev)
            kw = {"sid": torch.zeros(1, dtype=torch.long, device=self.dev), "lid": torch.zeros(1, dtype=torch.long, device=self.dev)}
            w, _, _ = self.G.infer(ids, torch.tensor([ids.size(1)], device=self.dev), **kw)
            sf.write(self.run / "samples" / f"step{self.step:07d}_{i}.wav", w[0, 0].float().cpu().numpy(), self.mc.sample_rate)
        self.G.train()

    def fit(self):
        t = self.cfg["train"]; accum = t.get("grad_accum", 1)
        sampler = BucketSampler(self.tr.frames, t["batch_size"], seed=t.get("seed", 0), languages=[r.get("language", "en") for r in self.tr.rows], balance=t.get("language_balanced", False))
        self.G.train(); self.D.train()
        while self.step < t["max_steps"]:
            sampler.set_epoch(self.epoch)
            dl = DataLoader(self.tr, batch_sampler=sampler, collate_fn=collate, num_workers=t.get("workers", 4), persistent_workers=False, prefetch_factor=2 if t.get("workers", 4) else None)
            t_ep = time.time(); it = iter(dl); exhausted = False
            while not exhausted and self.step < t["max_steps"]:
                t0 = time.perf_counter()
                gan = self.step >= t.get("gan_start_step", 0)
                self.oG.zero_grad(set_to_none=True); self.oD.zero_grad(set_to_none=True)
                out, got = {}, 0
                try:
                    for _ in range(accum):
                        try: b = next(it)
                        except StopIteration: exhausted = True; break
                        out = self.micro(b, accum, gan); got += 1
                except RuntimeError as e:   # e.g. MPS out-of-memory: log, free cache, skip this step
                    self.log(event="batch_error", step=self.step, error=str(e)[:300], mem_mb=memory_allocated_mb(self.dev))
                    if self.dev.type == "mps": torch.mps.empty_cache()
                    continue
                if got == 0: break
                self.scG.unscale_(self.oG); gn = torch.nn.utils.clip_grad_norm_(self.G.parameters(), t["clip_grad"])
                if gan:
                    self.scD.unscale_(self.oD); torch.nn.utils.clip_grad_norm_(self.D.parameters(), t["clip_grad"])
                    self.scD.step(self.oD); self.scD.update()
                self.scG.step(self.oG); self.scG.update(); self.step += 1
                dt = time.perf_counter() - t0
                if self.step % t["log_every"] == 0:
                    self.log(event="train", step=self.step, epoch=self.epoch, step_s=dt, grad_norm=float(gn), lr=self.oG.param_groups[0]["lr"],
                             mem_mb=memory_allocated_mb(self.dev), gan=gan, lang_mel=self.last_lang_mel, **out)
                    print(f"step {self.step} mel {out['mel']:.3f} kl {out['kl']:.3f} dur {out['dur']:.3f} {dt:.2f}s", flush=True)
                if self.step % t["val_every"] == 0:
                    self.log(event="val", step=self.step, **self.validate()); self.synth_samples()
                if self.step % t["ckpt_every"] == 0:
                    p = self.save(); self.log(event="ckpt", step=self.step, path=str(p), size_mb=p.stat().st_size / 2**20)
            self.epoch += 1; self.sG.step(); self.sD.step()
            self.log(event="epoch", epoch=self.epoch, seconds=time.time() - t_ep, step=self.step)
        p = self.save(); self.log(event="done", step=self.step, path=str(p))
