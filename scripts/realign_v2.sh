#!/bin/bash
# Re-align with the tail extension into <lang>/align_v2 (NEVER touches align/, which a running training reads). CPU only, one language at a time.
# Usage: scripts/realign_v2.sh [lang ...]; features are alignment-independent (verified: fs2_features has no alignment input) so they are not redone.
cd "$(dirname "$0")/.." || exit 1
export HF_HUB_DISABLE_XET=1
model_for() { case $1 in
  hi) echo "--model Harveenchadha/vakyansh-wav2vec2-hindi-him-4200 --license MIT";;
  bn) echo "--model shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice --license Apache-2.0";;
  ta) echo "--model Harveenchadha/vakyansh-wav2vec2-tamil-tam-250 --license MIT";;
  te) echo "--model anuragshas/wav2vec2-large-xlsr-53-telugu --license Apache-2.0";;
  mr) echo "--model sumedh/wav2vec2-large-xlsr-marathi --license Apache-2.0";;
  ml) echo "--model gvs/wav2vec2-large-xlsr-malayalam --license Apache-2.0";;
  pa) echo "--model Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10 --license MIT";;
  or) echo "--model theainerd/wav2vec2-large-xlsr-53-odia --license Apache-2.0";;
  as) echo "--model jayashreedevi2020/wav2vec2-large-xls-r-300m-assamese --license Apache-2.0";;
  ur) echo "--model kingabzpro/wav2vec2-large-xls-r-300m-Urdu --license Apache-2.0";;
  *) echo "";; esac; }
guard() { while :; do SW=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}'); FD=$(df -g . | tail -1 | awk '{print $4}')
  [ "$SW" -le 12288 ] && [ "$FD" -ge 10 ] && break; echo "waiting: swap ${SW}M disk ${FD}G"; sleep 60; done; }
for l in ${@:-de fr es it nl zh hi bn ta te pa ur mr as or ml}; do
  [ "$l" = en ] && continue
  guard; .venv/bin/python scripts/align_multilingual.py $l $(model_for $l) --align-dir align_v2 --tail-extend --keep-model
done
