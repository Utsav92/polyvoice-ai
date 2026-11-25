#!/bin/bash
# Re-runnable, CPU-only, one language at a time: align + extract FS2 features for language dirs under data/datasets/multi that have a manifest and a
# licence-compatible aligner. Skips finished utterances. Waits while swap used > 12 GB or free disk < 10 GB. Usage: scripts/prepare_multi.sh [lang ...]
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_DISABLE_XET=1
LANGS=${@:-"de fr es it nl"}     # zh done; en = existing LJSpeech alignment; Indic aligners are chosen per language (see docs/model_licenses.md), run with align_multilingual.py --model
guard() {
  while :; do
    SW=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'); FD=$(df -g . | tail -1 | awk '{print $4}')
    [ "$SW" -le 12288 ] && [ "$FD" -ge 10 ] && break; echo "waiting: swap ${SW}M disk ${FD}G"; sleep 60
  done
}
for l in $LANGS; do
  [ -s data/datasets/multi/$l/manifest.jsonl ] || continue
  guard; .venv/bin/python scripts/align_multilingual.py $l ${ALIGN_ARGS}
  guard; .venv/bin/python scripts/extract_fs2_features.py --workers 2 --multi-lang-dir data/datasets/multi/$l
done
