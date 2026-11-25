from __future__ import annotations
import gc, glob, json, os, time
from dataclasses import asdict
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from ml.inference.device import detect_device, memory_allocated_mb
from ml.models.fastspeech2.model import FastSpeech2, FS2Config, fs2_loss
from ml.training.fs2_data import FS2Dataset, fs2_collate, compute_stats, compute_stats_multi, language_coverage
from ml.preprocessing.tokenizer import CharTokenizer

class FS2Trainer:
    def __init__(self, cfg, resume=True):
        self.cfg, t = cfg, cfg["train"]; self.dev = detect_device(cfg.get("device"))
        self.run = Path(cfg["run_dir"]); (self.run / "ckpt").mkdir(parents=True, exist_ok=True)
        root = Path(cfg["data_root"]); mc_ = cfg.get("multi"); self.sampler_w = None
        # Opt-in multilingual mode. multi: {languages: [FULL fixed list, index = position], speaker_slots: 1024, min_coverage: 0.95, sampling_alpha: 0.5,
        #   tokenizer: path, min_score: float|null}. Embedding sizes depend only on that list and speaker_slots, so languages/speakers can be added on restart.
        if mc_:
            all_langs = list(mc_["languages"]); cov = {l: language_coverage(root / l) for l in all_langs}
            ready = [l for l in all_langs if cov[l]["ready"] > 0 and cov[l]["coverage"] >= mc_.get("min_coverage", 0.95)]
            dirs = [root / l for l in ready]; tokp = Path(mc_.get("tokenizer", root / "tokenizer.json"))
            sp = self.run / "fs2_stats_frozen.json"                       # computed ONCE (first start), never recomputed on restarts
            if not sp.exists(): sp.write_text(json.dumps(compute_stats_multi(dirs)))
            kw = dict(lang_dirs=dirs, tokenizer_path=tokp, min_score=mc_.get("min_score"))
            self.tr = FS2Dataset(root, "train", json.loads(sp.read_text()), all_langs, None, **kw)
            mp = self.run / "speaker_map.json"; smap = json.loads(mp.read_text()) if mp.exists() else {}      # speaker -> slot, append-only
            for sname in sorted({r.get("speaker", "spk") for r in self.tr.rows} - set(smap)): smap[sname] = len(smap)
            slots = mc_.get("speaker_slots", 1024)
            if len(smap) > slots: raise RuntimeError(f"{len(smap)} speakers exceed speaker_slots={slots}")
            mp.write_text(json.dumps(smap, indent=0, ensure_ascii=False)); order = [None] * len(smap)
            for sname, i in smap.items(): order[i] = sname
            self.tr.speakers = order; self.stats = json.loads(sp.read_text()); self.tok = CharTokenizer.load(tokp)
            self.va = FS2Dataset(root, "val", self.stats, all_langs, order, **kw)
            n_spk, n_lang = slots, len(all_langs)
            alpha = mc_.get("sampling_alpha", 0.5); hrs, cnt = {}, {}
            for r in self.tr.rows: l = r.get("language"); hrs[l] = hrs.get(l, 0) + r["duration_s"] / 3600; cnt[l] = cnt.get(l, 0) + 1
            z = sum(h ** alpha for h in hrs.values()); share = {l: hrs[l] ** alpha / z for l in hrs}
            self.sampler_w = [share[r.get("language")] / cnt[r.get("language")] for r in self.tr.rows]
            info = {"ready_languages": ready, "coverage": cov, "stats_computed_from": json.loads(sp.read_text()).get("languages"), "train_hours": hrs,
                    "sampling_alpha": alpha, "sampling_share": share, "n_speakers_assigned": len(smap), "language_index": {l: i for i, l in enumerate(all_langs)}}
            (self.run / "data_info.json").write_text(json.dumps(info, indent=1)); print(json.dumps({k: info[k] for k in ("ready_languages", "train_hours", "sampling_share")}), flush=True)
        else:
            sp = root / "fs2_stats.json"; tokp = root / "tokenizer.json"
            if not sp.exists(): sp.write_text(json.dumps(compute_stats(root)))
            self.stats = json.loads(sp.read_text()); self.tok = CharTokenizer.load(tokp)
            self.tr = FS2Dataset(root, "train", self.stats); self.va = FS2Dataset(root, "val", self.stats, self.tr.languages, self.tr.speakers)
            n_spk, n_lang = len(self.tr.speakers), len(self.tr.languages)
        self.mc = FS2Config(**cfg["model"], n_vocab=self.tok.vocab_size, n_speakers=n_spk, n_languages=n_lang)
        self.net = FastSpeech2(self.mc).to(self.dev)
        self.opt = torch.optim.AdamW(self.net.parameters(), t["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=t.get("wd", 0.0))
        w = t["warmup"]; self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min((s + 1) / w, (w / (s + 1)) ** 0.5))
        self.step = 0
        if resume: self.load()
        self.log_f = open(self.run / "train_log.jsonl", "a"); (self.run / "config.yaml").write_text(yaml.safe_dump(cfg))
    def save(self):
        p = self.run / "ckpt" / f"step_{self.step:07d}.pt"
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "step": self.step, "model_cfg": asdict(self.mc),
                    "stats": self.stats, "teacher": self.cfg.get("teacher"),
                    "languages": (self.cfg.get("multi") or {}).get("languages"), "speaker_map": json.loads((self.run / "speaker_map.json").read_text()) if (self.run / "speaker_map.json").exists() else None}, p)
        for o in sorted(glob.glob(str(self.run / "ckpt" / "step_*.pt")))[:-self.cfg["train"].get("keep_ckpts", 3)]: os.remove(o)
        return p
    def load(self):
        c = sorted(glob.glob(str(self.run / "ckpt" / "step_*.pt")))
        if not c: return
        s = torch.load(c[-1], map_location="cpu", weights_only=False); self.net.load_state_dict(s["net"]); self.opt.load_state_dict(s["opt"]); self.sched.load_state_dict(s["sched"]); self.step = s["step"]
        print("resumed", c[-1], self.step)
    def log(self, **kw): kw["time"] = time.time(); self.log_f.write(json.dumps(kw) + "\n"); self.log_f.flush()
    def _d(self, b): return {k: (v.to(self.dev) if torch.is_tensor(v) else v) for k, v in b.items()}
    def _loss(self, b):
        o = self.net(b["tokens"], b["tok_len"], b["dur"], b["pitch"], b["energy"], b["mel_len"], b["sid"], b["lid"])
        return fs2_loss(o, b["mel"], b["dur"], b["pitch"], b["energy"])
    @torch.no_grad()
    def validate(self, n=16):
        self.net.eval(); agg, k = {}, 0
        for i, b in enumerate(DataLoader(self.va, batch_size=8, collate_fn=fs2_collate)):
            if i >= n // 8 * 4: break
            _, parts = self._loss(self._d(b)); k += 1
            for a, v in parts.items(): agg[f"val_{a}"] = agg.get(f"val_{a}", 0) + v
        self.net.train(); return {a: v / max(k, 1) for a, v in agg.items()}
    def fit(self):
        t = self.cfg["train"]; self.net.train()
        consec_err = 0
        while self.step < t["max_steps"]:
            smp = dict(sampler=WeightedRandomSampler(self.sampler_w, num_samples=len(self.tr), replacement=True)) if self.sampler_w else dict(shuffle=True)
            dl = DataLoader(self.tr, batch_size=t["batch_size"], drop_last=True, collate_fn=fs2_collate, num_workers=t.get("workers", 2), **smp)
            for b in dl:
                t0 = time.perf_counter(); b = self._d(b)
                try:
                    self.opt.zero_grad(set_to_none=True); loss, parts = self._loss(b); loss.backward()
                except RuntimeError as e:
                    self.log(event="batch_error", step=self.step, error=str(e)[:200], mem_mb=memory_allocated_mb(self.dev))
                    del e; b = loss = parts = None; self.opt.zero_grad(set_to_none=True); gc.collect()   # drop tensors held by the failed step's frames so memory is really freed
                    if self.dev.type == "mps": torch.mps.empty_cache()
                    consec_err += 1
                    if consec_err > 20: raise RuntimeError("20 consecutive batch errors (out of memory?): aborting instead of looping; lower batch_size")
                    continue
                consec_err = 0
                gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), t["clip_grad"]); self.opt.step(); self.sched.step(); self.step += 1
                if self.step % t["log_every"] == 0:
                    self.log(event="train", step=self.step, loss=float(loss), step_s=time.perf_counter() - t0, grad_norm=float(gn), lr=self.sched.get_last_lr()[0], mem_mb=memory_allocated_mb(self.dev), **parts)
                    print(f"step {self.step} loss {float(loss):.3f} {parts}", flush=True)
                if self.step % t["val_every"] == 0: self.log(event="val", step=self.step, **self.validate())
                if self.step % t["ckpt_every"] == 0: p = self.save(); self.log(event="ckpt", step=self.step, size_mb=p.stat().st_size / 2**20)
                if self.step >= t["max_steps"]: break
        p = self.save(); self.log(event="done", step=self.step, path=str(p))
