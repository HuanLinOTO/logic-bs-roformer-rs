#!/usr/bin/env python3
"""Run the Cargo-built lbrr test harness without launching a production binary."""
import argparse,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('deps',type=Path);a=p.parse_args()
found=[]
# New Cargo build-directory layouts put harnesses under build/<crate>/<hash>/out.
candidates=list(a.deps.glob('lbrr-*'))+list(a.deps.parent.glob('build/lbrr/*/out/lbrr-*'))
for path in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
    if not path.is_file() or path.suffix not in ('','.exe') or not os.access(path,os.X_OK):continue
    probe=subprocess.run([str(path),'--list'],capture_output=True,text=True,timeout=30)
    if probe.returncode==0 and 'benchmark::tests::verified_flop_counts' in probe.stdout:
        found.append(path);break
if not found:raise SystemExit('No lbrr unit-test harness; build with cargo oxide build -- --release --tests')
subprocess.run([str(found[0]),'--nocapture','--test-threads=1'],check=True)
