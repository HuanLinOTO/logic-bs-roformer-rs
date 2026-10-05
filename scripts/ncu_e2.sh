#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
MET="sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_alu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_fmaheavy.sum,sm__inst_executed_pipe_fmalite.sum,smsp__inst_executed.sum,sm__sass_inst_executed_op_shared_ld.sum,sm__sass_inst_executed_op_shared_st.sum,sm__sass_inst_executed_op_global.sum"
for kn in gemm_f16_resid_a16 gemm_f16_128x64_hout gemm_f16_gelu_a16w; do
  /usr/bin/ncu --csv --metrics "$MET" -k "regex:$kn" --launch-count 1 --launch-skip 12 /data/dsh/target-lbrr/release/lbrr --bench --iters 30 --model-dir assets > /data/dsh/pipe2_$kn.csv 2>&1
  /usr/bin/ncu --csv --section SpeedOfLight -k "regex:$kn" --launch-count 1 --launch-skip 12 /data/dsh/target-lbrr/release/lbrr --bench --iters 30 --model-dir assets > /data/dsh/sol2_$kn.csv 2>&1
  echo "==== $kn pipes ===="
  awk -F'","' '$1 ~ /^"0/ && $13 ~ /^sm|^smsp/ {print $13" = "$15}' /data/dsh/pipe2_$kn.csv
  echo "---- SOL ----"
  awk -F'","' '$1 ~ /^"0/ && ($13 ~ /Throughput/ || $13 ~ /Duration/) {print $13" = "$15" "$14}' /data/dsh/sol2_$kn.csv | head -8
done
echo NCU-E2-DONE
