"""Forced alignment of characters to audio with an English CTC acoustic model (wav2vec2-base-960h, Apache-2.0).
Used only to produce duration LABELS for training; model weights trained downstream stay randomly initialised."""
from __future__ import annotations
import numpy as np
from numba import njit

NEG = -1e30

@njit(cache=True)
def _viterbi(logp, ext):
    T, S = logp.shape[0], len(ext)
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, ext[0]]
    if S > 1: dp[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], 0
            if s >= 1 and dp[t - 1, s - 1] > best: best, arg = dp[t - 1, s - 1], 1
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best: best, arg = dp[t - 1, s - 2], 2
            if best > NEG / 2: dp[t, s] = best + logp[t, ext[s]]; bp[t, s] = arg
    s = S - 1 if (S == 1 or dp[T - 1, S - 1] >= dp[T - 1, S - 2]) else S - 2
    states = np.zeros(T, dtype=np.int32)
    for t in range(T - 1, -1, -1):
        states[t] = s; s -= bp[t, s]
    return states, dp[T - 1, S - 1] if S == 1 else max(dp[T - 1, S - 1], dp[T - 1, S - 2])

def ctc_align(logp: np.ndarray, targets: list[int], blank: int = 0):
    """logp [T,C] log-probs; targets non-blank ids. Returns (start[L], end[L] frame indices, score). Raises if T too short."""
    L = len(targets)
    if logp.shape[0] < L: raise ValueError("fewer frames than targets")
    ext = np.zeros(2 * L + 1, dtype=np.int64); ext[1::2] = targets
    states, score = _viterbi(logp.astype(np.float64), ext)
    start = np.full(L, -1); end = np.full(L, -1)
    for t, s in enumerate(states):
        if s % 2 == 1:
            k = (s - 1) // 2
            if start[k] < 0: start[k] = t
            end[k] = t
    return start, end, float(score)

def _integerize(r: np.ndarray, total: int) -> np.ndarray:
    """Integer durations >= 1 that sum exactly to `total`, distributing the surplus in proportion to (r - 1)^+ (largest remainder)."""
    n = len(r)
    if total < n: raise ValueError(f"need at least {n} frames, got {total}")
    extra = np.maximum(np.asarray(r, dtype=np.float64) - 1.0, 0.0); spare = total - n
    if extra.sum() <= 0: extra = np.ones(n)
    share = extra / extra.sum() * spare; base = np.floor(share).astype(np.int64)
    rem = spare - base.sum()
    if rem > 0: base[np.argsort(-(share - base))[:rem]] += 1
    out = (1 + base).astype(np.int32); assert out.sum() == total and (out >= 1).all()
    return out

def char_durations(start, end, n_frames, with_blanks: bool, out_frames: int):
    """CTC spans (CTC frame rate) -> integer durations on an `out_frames` grid. Every token gets >= 1 frame, total is exact.
    Gaps after a char are absorbed by that char. with_blanks=True -> VITS sequence [b,c,b,c,...,b]: leading silence goes to the
    first blank, trailing silence to the last blank, inner blanks take ~1 frame. with_blanks=False -> one duration per char
    (leading silence folded into the first char, trailing into the last)."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    L = len(start); scale = out_frames / n_frames
    b = start * scale                                         # char k begins at b[k]
    stop = (end[-1] + 1) * scale                              # speech ends here; remainder is trailing silence
    nxt = np.append(b[1:], stop)
    if with_blanks:
        r = np.ones(2 * L + 1)
        r[0] = b[0] + 1.0; r[-1] = max(out_frames - stop, 0.0) + 1.0
        r[1::2] = np.maximum(nxt - b - 1.0, 1.0)
        return _integerize(r, out_frames)
    r = nxt - b; r[0] += b[0]; r[-1] += max(out_frames - stop, 0.0)
    return _integerize(r, out_frames)

def fs2_durations(start, end, n_frames, out_frames):
    """FastSpeech 2 layout: [sil, c1..cL, sil] with explicit leading/trailing silence tokens (id 0)."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    L = len(start); scale = out_frames / n_frames; b = start * scale; stop = (end[-1] + 1) * scale
    r = np.concatenate([[b[0]], np.append(b[1:], stop) - b, [max(out_frames - stop, 0.0)]]) 
    return _integerize(np.maximum(r, 1.0), out_frames)
