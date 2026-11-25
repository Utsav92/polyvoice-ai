from __future__ import annotations
import re, unicodedata
import numpy as np

def _edit(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1])); prev = cur
    return d[-1]

def clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("-", " ")
    s = "".join(c for c in s if unicodedata.category(c)[0] not in "PS")   # keeps combining marks (Hindi etc.)
    return re.sub(r"\s+", " ", s).strip()

def wer(ref: str, hyp: str) -> float:
    r, h = clean(ref).split(), clean(hyp).split()
    return _edit(r, h) / max(1, len(r))

def cer(ref: str, hyp: str) -> float:
    r, h = list(clean(ref)), list(clean(hyp))
    return _edit(r, h) / max(1, len(r))

def corpus(refs, hyps, fn):
    """Micro-average: total edits / total reference length."""
    tot = sum(_edit(list(clean(r).split()) if fn is wer else list(clean(r)), list(clean(h).split()) if fn is wer else list(clean(h))) for r, h in zip(refs, hyps))
    n = sum(len(clean(r).split()) if fn is wer else len(clean(r)) for r in refs)
    return tot / max(1, n)
