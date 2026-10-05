#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
ok=1
for f in separated2/*.wav; do
  b="separated_ldmx/$(basename "$f")"
  if cmp -s "$f" "$b"; then echo "SAME $(basename "$f")"; else echo "DIFF $(basename "$f")"; ok=0; fi
done
[ "$ok" = 1 ] && echo ALL-BITWISE-EQUAL || echo MISMATCH-FOUND
