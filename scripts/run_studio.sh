#!/bin/bash
# Start PolyVoice Studio: natural-voice worker (.venv-nat, port 8021) + studio server (.venv, port 8020), then open the browser.
cd "$(dirname "$0")/.."
pgrep -f "scripts/natural_worker.py" >/dev/null || (HF_HUB_DISABLE_XET=1 nohup .venv-nat/bin/python scripts/natural_worker.py --port 8021 > data/datasets/multi/_logs/natural_worker.log 2>&1 &)
pgrep -f "scripts/serve_multi_tts.py" >/dev/null || (nohup .venv/bin/python scripts/serve_multi_tts.py --port 8020 > data/datasets/multi/_logs/studio.log 2>&1 &)
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8020/api/info && break; sleep 2; done
open http://127.0.0.1:8020; echo "Studio: http://127.0.0.1:8020  (stop: pkill -f natural_worker.py; pkill -f serve_multi_tts.py)"
