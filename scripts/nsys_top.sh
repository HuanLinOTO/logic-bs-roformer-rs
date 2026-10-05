#!/bin/bash
cd /data/dsh/logic-bs-roformer-rs
rm -f /data/dsh/nsys_ldmx.nsys-rep /data/dsh/nsys_ldmx.qdstrm
/usr/local/bin/nsys profile -o /data/dsh/nsys_ldmx --force-overwrite=true /data/dsh/target-lbrr/release/lbrr --bench --iters 30 --model-dir assets 2>&1 | tail -4
/usr/local/bin/nsys stats --report cuda_gpu_kern_sum /data/dsh/nsys_ldmx.nsys-rep 2>/dev/null | head -28
echo NSYS-DONE
