#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
for s in 0_bass 1_drums 2_other 3_vocals 4_guitar 5_piano; do
  printf '%s : ' "$s"
  /data/dsh/lbrr-venv/bin/python /data/dsh/snr_wav.py "separated_final/$s.wav" "separated/$s.wav" 2>&1 | tail -1
done
