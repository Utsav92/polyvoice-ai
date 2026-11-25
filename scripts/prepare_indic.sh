#!/bin/bash
# CPU-only (macOS bash 3.2 compatible), one language at a time: full alignment with the aligner chosen by measured sample quality (docs/datasets.md), then FS2 features.
# Usage: scripts/prepare_indic.sh [lang ...]   (kn and gu are intentionally absent: no licence-verified model)
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_DISABLE_XET=1
model_for() { case $1 in
  hi) echo "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200 MIT";;
  bn) echo "shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice Apache-2.0";;
  ta) echo "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250 MIT";;
  te) echo "anuragshas/wav2vec2-large-xlsr-53-telugu Apache-2.0";;
  mr) echo "sumedh/wav2vec2-large-xlsr-marathi Apache-2.0";;
  ml) echo "gvs/wav2vec2-large-xlsr-malayalam Apache-2.0";;
  pa) echo "Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10 MIT";;
  or) echo "theainerd/wav2vec2-large-xlsr-53-odia Apache-2.0";;
  as) echo "jayashreedevi2020/wav2vec2-large-xls-r-300m-assamese Apache-2.0";;
  ur) echo "kingabzpro/wav2vec2-large-xls-r-300m-Urdu Apache-2.0";;
  *) echo "";; esac; }
guard() { while :; do SW=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'); FD=$(df -g . | tail -1 | awk '{print $4}')
  [ "$SW" -le 12288 ] && [ "$FD" -ge 10 ] && break; echo "waiting: swap ${SW}M disk ${FD}G"; sleep 60; done; }
for l in ${@:-hi bn ta te}; do
  MD=$(model_for $l); [ -n "$MD" ] || { echo "no aligner chosen for $l, skipping"; continue; }
  [ -s data/datasets/multi/$l/manifest.jsonl ] || continue
  set -- $MD
  guard; .venv/bin/python scripts/align_multilingual.py $l --model $1 --license $2 --keep-model ${ALIGN_ARGS}
  guard; .venv/bin/python scripts/extract_fs2_features.py --workers 2 --multi-lang-dir data/datasets/multi/$l
done
