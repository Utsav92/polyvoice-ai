#!/usr/bin/env python
"""PolyVoice launch-style demo video. Usage: build_demo.py [curated|raw] [--sheet] [--out demo.mp4]
Text is shaped by Core Text (./rendertext); frames are composited with Pillow and piped to ffmpeg. Audio = real studio TTS + an original synthesized ambient pad."""
import hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, SR = 1920, 1080, 30, 48000
ROOT = Path(__file__).parent; A = ROOT / "assets"; CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
VARIANT = next((a for a in sys.argv[1:] if a in ("curated", "raw")), "curated"); SHEET = "--sheet" in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"polyvoice_demo_{VARIANT}.mp4"
MAT = json.load(open(A / ("material.json" if VARIANT == "curated" else "material_raw.json")))
WAV = (lambda c: A / (f"{c}.wav" if VARIANT == "curated" else f"raw_{c}.wav"))
INPUT = MAT["input"]; ORDER = ["en", "de", "fr", "it", "es", "hi", "zh"]; LANGS = MAT["langs"]
NAMES = {"en": "English", "de": "Deutsch", "fr": "Français", "it": "Italiano", "es": "Español", "hi": "हिन्दी", "zh": "中文"}
ENGB = {"kokoro": "Kokoro · Apache-2.0", "piper": "Piper · CC0 voice data"}
BLUE, PURPLE, PINK, GRAY, LGRAY = (41, 151, 255), (175, 82, 222), (255, 85, 140), (142, 142, 147), (229, 229, 234)

# ------------------------------------------------------------------ helpers
clamp = lambda x, a=0.0, b=1.0: max(a, min(b, x))
def prog(t, t0, t1): return clamp((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t0)
def ease(x): return 1 - (1 - x) ** 3
def ease2(x): return 3 * x * x - 2 * x * x * x
def fade(t, a0, a1, b0, b1): return ease2(prog(t, a0, a1)) * (1 - ease2(prog(t, b0, b1)))   # in over [a0,a1], out over [b0,b1]

_raw = {}
def txt(text, size, weight="regular", width=1700, align="left", lh=1.2, tr=0.0):
    key = hashlib.md5(f"{text}|{size}|{weight}|{width}|{align}|{lh}|{tr}".encode()).hexdigest()
    if key not in _raw:
        p = CACHE / f"{key}.png"
        if not p.exists(): subprocess.run([str(ROOT / "rendertext"), str(p), text, str(size), weight, "FFFFFF", str(width), align, str(lh), str(tr)], check=True, capture_output=True)
        _raw[key] = Image.open(p).convert("RGBA")
    return _raw[key]
_tint = {}
def tinted(img, color):
    k = (id(img), tuple(color) if not callable(color) else "g")
    if k not in _tint:
        a = np.array(img); out = np.zeros_like(a); out[..., 3] = a[..., 3]
        if isinstance(color, tuple): out[..., :3] = color
        else: out[..., :3] = color(a.shape[1])      # gradient: callable(width) -> (1,w,3)
        _tint[k] = Image.fromarray(out)
    return _tint[k]
def gradient(stops):
    def f(w):
        x = np.linspace(0, 1, w)[None, :, None]; n = len(stops) - 1; out = np.zeros((1, w, 3))
        for c in range(3):
            out[0, :, c] = np.interp(x[0, :, 0], np.linspace(0, 1, n + 1), [s[c] for s in stops])
        return out.astype(np.uint8)
    return f
GRAD = gradient([BLUE, PURPLE, PINK])

def paste(cv, img, x, y, alpha=1.0, scale=1.0, center=False):
    if alpha <= 0.004: return
    if scale != 1.0: img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    if center: x -= img.width / 2; y -= img.height / 2
    if alpha < 1: img = img.copy(); img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
    x0, y0 = int(round(x)), int(round(y)); sx, sy = max(0, -x0), max(0, -y0); ex, ey = min(img.width, W - x0), min(img.height, H - y0)
    if ex > sx and ey > sy: cv.alpha_composite(img.crop((sx, sy, ex, ey)), (x0 + sx, y0 + sy))

def rounded(w, h, r, fill, border=None, ss=3):
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=fill)
    if border: d.rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, outline=border, width=ss)
    return im.resize((w, h), Image.LANCZOS)

def glow_field():
    gw, gh = W + 500, H + 500; yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32); out = np.zeros((gh, gw, 3), np.float32)
    for (cx, cy, r, col) in ((0.22, 0.2, 0.55, (30, 100, 255)), (0.85, 0.85, 0.6, (140, 60, 220)), (0.7, 0.1, 0.35, (255, 70, 140))):
        d = np.sqrt(((xx - cx * gw) / (r * gw)) ** 2 + ((yy - cy * gh) / (r * gw)) ** 2); k = np.clip(1 - d, 0, 1) ** 2.2
        out += k[..., None] * np.array(col, np.float32)
    return np.clip(out, 0, 255)
GLOW = glow_field()
def background(t, intensity):
    ox = int(250 + 120 * math.sin(t * 0.18)); oy = int(250 + 90 * math.cos(t * 0.14))
    a = (GLOW[oy:oy + H, ox:ox + W] * (intensity * 0.55)).astype(np.uint8)
    return Image.fromarray(np.dstack([a, np.full((H, W), 255, np.uint8)]), "RGBA")

# ------------------------------------------------------------------ audio, timeline
def load48(path):
    tmp = CACHE / (Path(path).stem + "_48k.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", str(SR), "-ac", "1", str(tmp)], check=True)
    y, _ = sf.read(tmp, dtype="float32"); rms = np.sqrt(np.mean(y ** 2)) + 1e-9
    y = y * (0.085 / rms); peak = np.max(np.abs(y)); return (y * min(1.0, 0.89 / peak)).astype(np.float32)
SEG = {c: load48(WAV(c)) for c in ORDER}; DUR = {c: len(SEG[c]) / SR for c in ORDER}
T_TYPE0, T_TYPE1 = 16.3, 20.2
t = 21.4; START = {}
for c in ORDER: START[c] = t; t += DUR[c] + 1.15
T_FINALE = t + 0.1; T_END = T_FINALE + 5.4; TOTAL = T_END + 5.6

def envelope(c):
    y = SEG[c]; n = int(len(y) / SR * FPS) + 2; e = np.array([np.sqrt(np.mean(y[int(i * SR / FPS):int((i + 1) * SR / FPS)] ** 2) + 1e-12) if len(y[int(i * SR / FPS):int((i + 1) * SR / FPS)]) else 1e-6 for i in range(n)])
    e = e / (np.percentile(e, 95) + 1e-9); e = np.clip(e, 0, 1.15); return np.convolve(e, [.25, .5, .25], mode="same")
ENV = {c: envelope(c) for c in ORDER}

def build_audio():
    n = int(TOTAL * SR); mix = np.zeros(n, np.float32); duck = np.ones(n, np.float32)
    for c in ORDER:
        s = int(START[c] * SR); mix[s:s + len(SEG[c])] += SEG[c]
        a, b = int((START[c] - 0.25) * SR), int((START[c] + DUR[c] + 0.5) * SR); duck[a:b] = 0.42
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    tt = np.arange(n) / SR; pad = np.zeros(n, np.float32)
    for f, g in ((73.4, .5), (110.0, .45), (146.8, .35), (220.0, .25), (277.2, .2), (329.6, .16), (440.0, .08)):      # D-major add9 pad, slow swells
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.05 + 0.013 * f / 100) * tt + f)
        pad += (g * lfo * np.sin(2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 0.11 * tt))).astype(np.float32)
    pad *= 0.07 / (np.max(np.abs(pad)) + 1e-9); pad *= np.clip(tt / 3.0, 0, 1) * np.clip((TOTAL - tt) / 3.5, 0, 1) * duck
    def bell(t0, f=880, amp=0.09):
        s = int(t0 * SR); k = np.arange(int(2.6 * SR)) / SR
        w = amp * (np.sin(2 * np.pi * f * k) + .5 * np.sin(2 * np.pi * f * 1.5 * k) + .25 * np.sin(2 * np.pi * f * 2.01 * k)) * np.exp(-k * 2.2) * np.clip(k / 0.01, 0, 1)
        mix[s:s + len(w)] += w.astype(np.float32)
    bell(9.85, 880); bell(T_FINALE + 0.05, 1174.7, .07)
    out = np.clip(mix + pad, -0.97, 0.97); sf.write(CACHE / "mix.wav", np.stack([out, out], 1), SR); return CACHE / "mix.wav"

# ------------------------------------------------------------------ static UI pieces
WIN = (160, 120, 1600, 820); LX, RX, PY, PW, PH = 200, 980, 210, 740, 690
WINDOW = rounded(WIN[2], WIN[3], 36, (28, 28, 30, 235), (255, 255, 255, 28))
PANE = rounded(PW, PH, 28, (255, 255, 255, 14), (255, 255, 255, 22))
CH_N = {c: rounded(int(txt(NAMES[c], 28, "medium").width + 44), 58, 29, (255, 255, 255, 26)) for c in ORDER[1:]}
CH_S = {c: rounded(int(txt(NAMES[c], 28, "medium").width + 44), 58, 29, (245, 245, 247, 255)) for c in ORDER[1:]}
BTN = rounded(PW - 64, 66, 33, (10, 132, 255, 255)); PLAY = rounded(76, 76, 38, (10, 132, 255, 255))
CHIP_POS = {}; x = LX + 32; y = 676
for c in ORDER[1:]:
    w = CH_N[c].width
    if x + w > LX + PW - 32: x = LX + 32; y += 72
    CHIP_POS[c] = (x, y); x += w + 12

def lbl(s, size=19, col=GRAY, tr=0.09): return tinted(txt(s, size, "semibold", 700, "left", 1.0, tr), col)
def draw_logo(cv, cx, cy, t, a=1.0, s=1.0):
    n, bw, gap = 9, 14, 11
    for i in range(n):
        hgt = (22 + 56 * (0.5 + 0.5 * math.sin(t * 2.4 + i * 0.7)) * (0.55 + 0.45 * math.sin(i * 0.9 + 1))) * s; x = cx + (i - (n - 1) / 2) * (bw + gap) * s
        col = tuple(int(BLUE[k] + (PINK[k] - BLUE[k]) * i / (n - 1)) for k in range(3)); bar = rounded(max(2, int(bw * s)), max(4, int(hgt)), int(bw * s / 2), col + (255,))
        paste(cv, bar, x - bw * s / 2, cy - hgt / 2, a)

def waveform(cv, c, t, x, y, wpx, hpx, a):
    n = 46; f = (t - START[c]) * FPS; env = ENV[c]; cols = [tuple(int(BLUE[k] + (PURPLE[k] - BLUE[k]) * i / (n - 1)) for k in range(3)) for i in range(n)]
    ss = 2; im = Image.new("RGBA", (wpx * ss, hpx * ss), (0, 0, 0, 0)); d = ImageDraw.Draw(im); bw = wpx * ss / n
    for i in range(n):
        idx = f - (n - 1 - i) * 0.9; v = 0.06 if idx < 0 or idx >= len(env) else float(env[int(idx)])
        if t > START[c] + DUR[c]: v = 0.06
        hh = max(8 * ss, v * (hpx * ss - 8)); fillc = cols[i] + (int(255 * (0.35 + 0.65 * i / (n - 1))),)
        d.rounded_rectangle((i * bw + bw * .2, (hpx * ss - hh) / 2, i * bw + bw * .8, (hpx * ss + hh) / 2), bw * .3, fill=fillc)
    paste(cv, im.resize((wpx, hpx), Image.LANCZOS), x, y, a)

def typed_prefix(t):
    p_ = clamp((t - T_TYPE0) / (T_TYPE1 - T_TYPE0)); n = len(INPUT) if p_ >= 1 else (int(p_ * len(INPUT)) // 3) * 3; return INPUT[:n]

def ui(cv, t, a):
    paste(cv, WINDOW, WIN[0], WIN[1], a)
    for i, col in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
        dot = rounded(22, 22, 11, col + (255,)); paste(cv, dot, WIN[0] + 36 + i * 32, WIN[1] + 30, a)
    paste(cv, tinted(txt("PolyVoice Studio", 22, "medium", 600, "center", 1.0), GRAY), 960, WIN[1] + 40, a, center=True)
    paste(cv, PANE, LX, PY, a); paste(cv, PANE, RX, PY, a)
    paste(cv, lbl("TEXT"), LX + 32, PY + 30, a); paste(cv, lbl("TRANSLATE TO"), LX + 32, 636, a)
    shown = INPUT if t >= T_TYPE1 else typed_prefix(t)
    if shown: paste(cv, tinted(txt(shown, 37, "regular", 676, "left", 1.25, -0.01), (255, 255, 255)), LX + 32, PY + 74, a)
    cur = next((c for c in ORDER if START[c] - 0.15 <= t < START[c] + DUR[c] + 1.15), None)
    for c in ORDER[1:]:
        paste(cv, (CH_S if c == cur else CH_N)[c], *CHIP_POS[c], a); ti = txt(NAMES[c], 28, "medium")
        paste(cv, tinted(ti, (29, 29, 31) if c == cur else (245, 245, 247)), CHIP_POS[c][0] + (CH_N[c].width - ti.width) / 2, CHIP_POS[c][1] + (58 - ti.height) / 2, a)
    pulse = 1 - 0.035 * math.sin(math.pi * prog(t, START[cur], START[cur] + 0.45)) if cur else 1.0
    bx, by = LX + 32, 814
    paste(cv, BTN, bx + (PW - 64) / 2, by + 33, a, pulse, center=True); paste(cv, tinted(txt("Generate speech", 27, "semibold", 600, "center"), (255, 255, 255)), bx + (PW - 64) / 2, by + 33, a, pulse, center=True)
    paste(cv, lbl("OUTPUT"), RX + 32, PY + 30, a)
    if cur is None:
        fin = t > START[ORDER[-1]]
        paste(cv, tinted(txt("Your audio appears here.", 30, "regular", 600, "center"), (99, 99, 102)), RX + PW / 2, PY + PH / 2 - 6, a, center=True)
        return
    L = LANGS[cur]; ts = t - START[cur]; k = ease(prog(ts, 0.0, 0.55)); aa = a * k; dy = (1 - k) * 22
    paste(cv, tinted(txt(NAMES[cur], 78, "bold", 676, "left", 1.0, -0.02), (255, 255, 255)), RX + 32, PY + 64 + dy, aa)
    paste(cv, lbl(ENGB.get(L["engine"], L["engine"]), 20, GRAY, 0.02), RX + 32, PY + 168 + dy, aa)
    head = "ORIGINAL TEXT" if cur == "en" else (f"TRANSLATED FROM ENGLISH TO {NAMES[cur].upper()}" if VARIANT == "curated" else f"MACHINE TRANSLATION, UNEDITED  ·  ENGLISH TO {NAMES[cur].upper()}")
    if cur in ("hi", "zh") and cur != "en": head = head.replace(NAMES[cur].upper(), {"hi": "HINDI", "zh": "CHINESE"}[cur])
    paste(cv, lbl(head, 17, GRAY, 0.06), RX + 32, PY + 226 + dy, aa)
    paste(cv, tinted(txt(L["spoken"], 33, "regular", 676, "left", 1.3, -0.005), (245, 245, 247)), RX + 32, PY + 262 + dy, aa)
    pr = ease(prog(ts, 0.2, 0.8)); paste(cv, PLAY, RX + 32 + 38, PY + 596, a * pr, center=True)
    playing = ts < DUR[cur]
    ic = ImageDraw.Draw(img := Image.new("RGBA", (76, 76), (0, 0, 0, 0)))
    if playing: ic.rounded_rectangle((26, 22, 34, 54), 3, fill="white"); ic.rounded_rectangle((42, 22, 50, 54), 3, fill="white")
    else: ic.polygon([(29, 21), (29, 55), (55, 38)], fill="white")
    paste(cv, img, RX + 32 + 38, PY + 596, a * pr, center=True)
    waveform(cv, cur, t, RX + 140, PY + 548, PW - 172, 96, a * pr)
    m = f"0:{int(min(ts, DUR[cur])):02d} / 0:{int(DUR[cur]):02d}"; paste(cv, tinted(txt(m, 19, "regular", 300, "right", 1.0), GRAY), RX + PW - 32 - 300 + 8, PY + 636, a * pr)

# ------------------------------------------------------------------ frame
def frame(t):
    inten = 1.0 if t < 15.5 else 0.55; cv = background(t, inten)
    if t < 4.2:
        k = fade(t, 0.3, 1.1, 3.3, 4.1); paste(cv, tinted(txt("Hello.", 230, "bold", 1700, "center", 1.0, -0.03), (255, 255, 255)), 960, 540, k, 0.94 + 0.06 * ease(prog(t, 0.3, 1.8)), center=True)
    elif t < 9.6:
        paste(cv, tinted(txt("Write in English.", 128, "semibold", 1700, "center", 1.0, -0.03), (255, 255, 255)), 960, 470 + 16 * (1 - ease(prog(t, 4.3, 5.1))), fade(t, 4.3, 5.1, 8.8, 9.5), center=True)
        paste(cv, tinted(txt("Hear it in their language.", 128, "semibold", 1700, "center", 1.0, -0.03), GRAD), 960, 640 + 16 * (1 - ease(prog(t, 5.9, 6.7))), fade(t, 5.9, 6.7, 8.8, 9.5), center=True)
    elif t < 15.2:
        a = fade(t, 9.8, 10.7, 14.4, 15.1); draw_logo(cv, 960, 330, t, a, 1.15)
        paste(cv, tinted(txt("PolyVoice", 210, "bold", 1700, "center", 1.0, -0.035), GRAD), 960, 520 + 14 * (1 - ease(prog(t, 10.0, 10.9))), a, center=True)
        paste(cv, tinted(txt("Say it in their language.", 58, "light", 1700, "center", 1.0, -0.01), GRAY), 960, 690, fade(t, 11.4, 12.2, 14.4, 15.1), center=True)
    elif t < T_FINALE:
        a = ease2(prog(t, 15.1, 16.0)) * (1 - ease2(prog(t, T_FINALE - 0.5, T_FINALE))); sc = 0.965 + 0.035 * ease(prog(t, 15.1, 16.2))
        sub = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ui(sub, t, 1.0)
        paste(cv, sub.resize((int(W * sc), int(H * sc)), Image.BILINEAR), W / 2, H / 2 + 8, a, center=True)
    elif t < T_END:
        names = ORDER[1:]; pos = [(450, 380), (960, 380), (1470, 380), (450, 560), (960, 560), (1470, 560)]
        for i, c in enumerate(names):
            k = fade(t, T_FINALE + 0.2 + i * 0.28, T_FINALE + 0.9 + i * 0.28, T_END - 0.7, T_END)
            paste(cv, tinted(txt(NAMES[c], 104, "semibold", 700, "center", 1.0, -0.02), (255, 255, 255)), pos[i][0], pos[i][1] + 14 * (1 - k), k, center=True)
        paste(cv, tinted(txt("Write once. Hear it in six languages.", 66, "medium", 1700, "center", 1.0, -0.02), GRAD), 960, 800, fade(t, T_FINALE + 2.1, T_FINALE + 2.9, T_END - 0.7, T_END), center=True)
    else:
        a = fade(t, T_END + 0.1, T_END + 1.0, TOTAL - 1.3, TOTAL - 0.3); draw_logo(cv, 960, 330, t, a, 1.0)
        paste(cv, tinted(txt("PolyVoice Studio", 132, "bold", 1700, "center", 1.0, -0.03), (255, 255, 255)), 960, 500, a, center=True)
        small = ("Translations in this demo were written by hand. Voices are open models (Kokoro, Piper) running locally." if VARIANT == "curated"
                 else "Machine translation shown unedited. Voices are open models (Kokoro, Piper) running locally.")
        paste(cv, tinted(txt(small, 27, "regular", 1500, "center", 1.3), GRAY), 960, 640, fade(t, T_END + 0.9, T_END + 1.7, TOTAL - 1.3, TOTAL - 0.3), center=True)
    return cv.convert("RGB")

if __name__ == "__main__":
    print(f"variant={VARIANT} total={TOTAL:.1f}s finale@{T_FINALE:.1f}s starts:", {c: round(START[c], 1) for c in ORDER}, flush=True)
    if SHEET:
        times = [1.5, 5.6, 7.5, 11.5, 15.8, 18.5, START["en"] + 2.5, START["de"] + 3, START["hi"] + 4, START["zh"] + 3, T_FINALE + 3.5, T_END + 3]
        tiles = [frame(x).resize((640, 360), Image.LANCZOS) for x in times]; sheet = Image.new("RGB", (640 * 3, 360 * 4))
        for i, im in enumerate(tiles): sheet.paste(im, ((i % 3) * 640, (i // 3) * 360))
        sheet.save(ROOT / f"sheet_{VARIANT}.png"); print("sheet saved"); sys.exit()
    mix = build_audio(); n = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(mix),
                           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{TOTAL:.3f}", "-movflags", "+faststart", str(ROOT / OUT)], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(frame(i / FPS).tobytes())
        if i % 150 == 0: print(f"frame {i}/{n}", flush=True)
    ff.stdin.close(); ff.wait(); print("DONE", OUT, f"{TOTAL:.1f}s", flush=True)
