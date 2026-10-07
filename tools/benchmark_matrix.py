#!/usr/bin/env python3
"""Prepare fixed fixtures, alternate paired processes and enforce numerical gates."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path
import numpy as np

SAMPLES=588800

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()

def source_hash(root):
    h=hashlib.sha256()
    paths=sorted(list((root/'src').glob('*.rs'))+[root/'Cargo.toml',root/'Cargo.lock',root/'rust-toolchain.toml'])
    for p in paths:
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()

def metrics(actual,reference):
    if actual.shape!=reference.shape: raise ValueError(f'shape mismatch {actual.shape} != {reference.shape}')
    a=actual.astype(np.float64).reshape(6,-1);r=reference.astype(np.float64).reshape(6,-1)
    results=[]
    for av,rv in zip(a,r):
        finite=bool(np.isfinite(av).all() and np.isfinite(rv).all())
        rms=float(np.sqrt(np.mean(rv*rv)));err=av-rv
        results.append({'finite':finite,'reference_rms':rms,'snr_db':float(10*np.log10(np.sum(rv*rv)/max(np.sum(err*err),1e-300))) if finite and rms>=1e-8 else None,'max_abs_error':float(np.max(np.abs(err))) if finite else None,'bit_equal':bool(np.array_equal(av,rv))})
    return {'finite':all(s['finite'] for s in results),'shape':list(actual.shape),'stems':results}

def numerical_gate(base,candidate):
    errors=[]
    for i,(b,c) in enumerate(zip(base['stems'],candidate['stems'])):
        if not c['finite']: errors.append(f'stem {i}: nonfinite');continue
        if b['reference_rms']<1e-8:
            if c['max_abs_error']>1e-6: errors.append(f'stem {i}: silence error')
        elif c['snr_db']<b['snr_db']-0.1 or c['max_abs_error']>max(1.1*b['max_abs_error'],1e-6):
            errors.append(f'stem {i}: regression against B0')
    return errors

def prepare(a):
    import torch
    from bench_ref import load_model
    torch.set_num_threads(8)
    a.out.mkdir(parents=True,exist_ok=True)
    old=np.load(a.model_dir/'ref_output.npz')
    with wave.open(str(a.audio)) as w:
        if w.getnchannels()!=2 or w.getsampwidth()!=2 or w.getframerate()!=44100: raise ValueError('fixture audio must be stereo 44.1kHz PCM16')
        song=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32).reshape(-1,2).T.copy()/32768.0
    if song.shape[-1]<SAMPLES: raise ValueError('fixture audio must contain one full chunk')
    starts={'full':0,'middle':(song.shape[-1]-SAMPLES)//2,'end':song.shape[-1]-SAMPLES}
    inputs={'short':old['inp'].reshape(1,2,-1)}
    inputs.update({k:song[:,v:v+SAMPLES][None].copy() for k,v in starts.items()})
    inputs['silence']=np.zeros((1,2,SAMPLES),np.float32)
    asym=inputs['full'].copy();asym[:,1]=np.roll(asym[:,1],317)*0.125;inputs['asymmetric']=asym
    for name,ch in [('impulse-left',0),('impulse-right',1)]:
        impulse=np.zeros((1,2,18944),np.float32);impulse[0,ch,[0,512,1024,18943]]=1;inputs[name]=impulse
    model=load_model(a.model_dir,a.reference_root)
    manifest={'schema_version':1,'sample_rate':44100,'model_sha256':sha(a.model_dir/'model.safetensors'),'config_sha256':sha(a.model_dir/'logic_bs_roformer.yaml'),'reference_root':str(a.reference_root.resolve()),'precision':'fp32','tf32':False,'torch':torch.__version__,'cases':{}}
    for name,inp in inputs.items():
        p=a.out/name;p.mkdir(exist_ok=True)
        if name=='short': out=old['out'].reshape(1,6,2,-1)
        else:
            with torch.inference_mode(): out=model(torch.from_numpy(inp).cuda()).float().cpu().numpy()
        if out.shape!=(1,6,2,inp.shape[-1]) or not np.isfinite(out).all(): raise RuntimeError(f'{name}: invalid reference')
        ref=p/'ref_output.npz';np.savez(ref,inp=inp.astype('<f4'),out=out.astype('<f4'))
        manifest['cases'][name]={'reference':str(ref.relative_to(a.out)),'sha256':sha(ref),'samples':inp.shape[-1],'T':inp.shape[-1]//512+1,'shape':list(out.shape),'finite':True}
        print('PREPARED',name,flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PREPARE_DONE',flush=True)

def coefficient(values):
    mean=statistics.mean(values)
    return statistics.pstdev(values)/mean if mean else 0.0

def run(a):
    manifest=json.loads((a.fixtures/'manifest.json').read_text())
    cases=a.cases.split(',') if a.cases else list(manifest['cases'])
    a.out.mkdir(parents=True,exist_ok=True)
    binaries={'B0':a.rust_bin.resolve()}
    if a.candidate_bin: binaries['candidate']=a.candidate_bin.resolve()
    elif not a.reference_root: raise ValueError('provide --candidate-bin or --reference-root')
    extra={'B0':json.loads(a.baseline_args),'candidate':json.loads(a.candidate_args)}
    if not all(isinstance(v,list) and all(isinstance(x,str) for x in v) for v in extra.values()): raise ValueError('extra args must be JSON string arrays')
    rows=[];telemetry=(a.out/'gpu-telemetry.csv').open('w');sampler=None
    try:
        sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used','--format=csv','--loop-ms=200'],stdout=telemetry,stderr=subprocess.STDOUT)
        for case in cases:
            ref=a.fixtures/manifest['cases'][case]['reference']
            if sha(ref)!=manifest['cases'][case]['sha256']: raise ValueError(f'fixture changed: {case}')
            for round_id in range(a.rounds):
                order=['B0','candidate' if a.candidate_bin else 'pytorch']
                if round_id%2: order.reverse()
                for label in order:
                    stem=f'{case}-{round_id}-{label}';report=a.out/(stem+'.json');log=a.out/(stem+'.log')
                    env=dict(os.environ)
                    if a.source_root: env['LBRR_SOURCE_SHA256']=source_hash(a.source_root)
                    if label=='pytorch':
                        cmd=[sys.executable,str(Path(__file__).with_name('bench_ref.py')),'--reference-root',str(a.reference_root),'--model-dir',str(a.model_dir),'--bench-ref',str(ref),'--warmup',str(a.warmup),'--iters',str(a.iters),'--json',str(report)]
                    else:
                        cmd=[str(binaries[label]),'--bench','--model-dir',str(a.model_dir),'--bench-ref',str(ref),'--bench-stage','waveform','--warmup',str(a.warmup),'--iters',str(a.iters),'--bench-json',str(report),'--bench-output',str(a.out/(stem+'.f32'))]+extra[label]
                    started=time.perf_counter()
                    with log.open('w') as f: p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
                    if p.returncode: raise RuntimeError(f'{stem}: exit {p.returncode}; see {log}')
                    result=json.loads(report.read_text());ms=statistics.median(result['timing_ms']['cuda_event'])
                    row={'case':case,'round':round_id,'label':label,'event_ms':ms,'host_ms':result['timing_ms']['host_per_iteration'],'process_ms':(time.perf_counter()-started)*1000,'report':report.name}
                    rows.append(row);(a.out/'runs.json').write_text(json.dumps(rows,indent=2));print('RESULT',json.dumps(row),flush=True)
        summary={}
        for case in cases:
            rr=[r for r in rows if r['case']==case];base=[r['event_ms'] for r in rr if r['label']=='B0'];cand=[r['event_ms'] for r in rr if r['label']!='B0']
            summary[case]={'b0_median_ms':statistics.median(base),'candidate_median_ms':statistics.median(cand),'b0_cv':coefficient(base),'candidate_cv':coefficient(cand),'stable':max(coefficient(base),coefficient(cand))<=0.02,'relative_improvement':1-statistics.median(cand)/statistics.median(base)}
            if a.candidate_bin:
                errors=[];directions=[];bit_equal=True
                ref=np.load(a.fixtures/manifest['cases'][case]['reference'])['out'];shape=ref.shape
                for i in range(a.rounds):
                    b=np.fromfile(a.out/f'{case}-{i}-B0.f32',dtype='<f4').reshape(shape);c=np.fromfile(a.out/f'{case}-{i}-candidate.f32',dtype='<f4').reshape(shape)
                    errors+=numerical_gate(metrics(b,ref),metrics(c,ref));bit_equal &= b.tobytes()==c.tobytes()
                    rb=next(r for r in rr if r['round']==i and r['label']=='B0');rc=next(r for r in rr if r['round']==i and r['label']=='candidate');directions.append(rc['event_ms']<rb['event_ms'])
                summary[case].update(numerical_errors=errors,bit_equal=bit_equal,faster_rounds=sum(directions))
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
        if any(s.get('numerical_errors') for s in summary.values()): raise RuntimeError('numerical acceptance failed')
    finally:
        if sampler:
            sampler.terminate()
            try: sampler.wait(timeout=10)
            except subprocess.TimeoutExpired: sampler.kill();sampler.wait()
        telemetry.close()

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--rust-bin',type=Path,required=True);r.add_argument('--candidate-bin',type=Path);r.add_argument('--reference-root',type=Path);r.add_argument('--model-dir',type=Path,default=Path('assets'));r.add_argument('--fixtures',type=Path,required=True);r.add_argument('--out',type=Path,required=True);r.add_argument('--rounds',type=int,default=5);r.add_argument('--iters',type=int,default=20);r.add_argument('--warmup',type=int,default=5);r.add_argument('--cases');r.add_argument('--source-root',type=Path);r.add_argument('--baseline-args',default='[]');r.add_argument('--candidate-args',default='[]')
    a=ap.parse_args()
    if a.action=='run' and (a.rounds<1 or a.iters<1 or a.warmup<0): ap.error('invalid round/iteration/warmup count')
    (prepare if a.action=='prepare' else run)(a)
if __name__=='__main__': main()
