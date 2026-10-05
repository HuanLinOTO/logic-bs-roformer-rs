#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
/data/dsh/target-lbrr/release/lbrr --separate --model-dir assets --input assets/cyberangel.wav --outdir separated_ldmx2 2>&1 | grep -E "demix|panic"
ok=1
for f in separated2/*.wav; do
  b="separated_ldmx2/$(basename "$f")"
  if cmp -s "$f" "$b"; then echo "SAME $(basename "$f")"; else echo "DIFF $(basename "$f")"; ok=0; fi
done
[ "$ok" = 1 ] && echo ALL-BITWISE-EQUAL || echo MISMATCH-FOUND
