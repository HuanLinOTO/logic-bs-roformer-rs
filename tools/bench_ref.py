#!/usr/bin/env python3
"""Fixed-input PyTorch waveform benchmark. GPU-resident input/output, no TF32."""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

CHUNK, STEP, BORDER = 588800, 559360, 29440

def load_model(model_dir, reference_root=None):
    root=Path(reference_root or Path(__file__).resolve().parent.parent)
    sys.path.insert(0,str(root))
    from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
    from safetensors.torch import load_file
    with open(Path(model_dir)/'logic_bs_roformer.yaml') as f:
        cfg=yaml.unsafe_load(f)
    mcfg=dict(cfg['model'])
    for key in ['multi_stft_resolution_loss_weight','multi_stft_resolutions_window_sizes','multi_stft_hop_size','multi_stft_normalized','linear_transformer_depth','use_torch_checkpoint','dim_freqs_in']:
        mcfg.pop(key,None)
    model=BSRoformer(**mcfg)
    missing,unexpected=model.load_state_dict(load_file(str(Path(model_dir)/'model.safetensors')),strict=False)
    if missing or unexpected:
        raise ValueError(f'checkpoint keys: missing={missing}, unexpected={unexpected}')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return model.eval().cuda()

def chunk_starts(length):
    if length<=0: raise ValueError('empty audio is unsupported')
    needed=BORDER+length
    last=max(0,(needed-CHUNK+STEP-1)//STEP)
    return [i*STEP for i in range(last+1)]

def reflect_pad(x,left,right):
    length=x.shape[-1]
    if not length: raise ValueError('empty audio is unsupported')
    if length==1: return x.expand(*x.shape[:-1],left+length+right)
    p=torch.arange(-left,length+right,device=x.device).remainder(2*(length-1))
    p=torch.where(p<length,p,2*(length-1)-p)
    return x.index_select(-1,p)

@torch.inference_mode()
def demix(model,x):
    length=x.shape[-1]; starts=chunk_starts(length); needed=BORDER+length
    xp=reflect_pad(x,BORDER,starts[-1]+CHUNK-needed)
    fade=torch.ones(CHUNK,device=x.device)
    fade[:BORDER]=torch.arange(BORDER,device=x.device,dtype=torch.float32)/BORDER
    fade[-BORDER:]=fade[:BORDER].flip(0)
    result=None;counter=torch.zeros(needed,device=x.device)
    for i,start in enumerate(starts):
        y=model(xp[...,start:start+CHUNK])
        if result is None: result=torch.zeros(*y.shape[:-1],needed,device=x.device,dtype=torch.float32)
        window=fade.clone()
        if i==0: window[:BORDER]=1
        if i==len(starts)-1: window[-BORDER:]=1
        take=min(CHUNK,needed-start)
        result[...,start:start+take]+=y[...,:take]*window[:take]
        counter[start:start+take]+=window[:take].square()
    if not bool((counter[BORDER:needed]>0).all()): raise RuntimeError('uncovered audio tail')
    return (result/counter.clamp_min(1e-8))[...,BORDER:needed]

def synth(sr,duration):
    rs=np.random.RandomState(0);t=np.arange(int(sr*duration))/sr
    x=np.stack([0.3*np.sin(2*np.pi*440*t)+0.2*np.sin(2*np.pi*880*t),0.25*np.sin(2*np.pi*443*t)+0.2*np.sin(2*np.pi*1760*t)]).astype(np.float32)
    return x+rs.randn(*x.shape).astype(np.float32)*0.01

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',type=Path,default=Path('assets'))
    ap.add_argument('--reference-root',type=Path)
    ap.add_argument('--bench-ref',type=Path)
    ap.add_argument('--full',action='store_true')
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--iters',type=int,default=20)
    ap.add_argument('--mode',choices=['fp32','amp'],default='fp32')
    ap.add_argument('--sdpa',choices=['auto','cudnn','flash'],default='auto')
    ap.add_argument('--json',type=Path)
    a=ap.parse_args()
    if a.iters<1 or a.warmup<0: ap.error('iters must be positive and warmup nonnegative')
    torch.set_num_threads(8)
    init=time.perf_counter();model=load_model(a.model_dir,a.reference_root)
    data=np.load(a.bench_ref or a.model_dir/'ref_output.npz')
    inp=data['inp'].reshape(1,2,-1) if not a.full else synth(44100,30)[None]
    if inp.shape[-1]<=1024 and not a.full: raise ValueError('single block requires L > n_fft/2')
    x=torch.from_numpy(inp.copy()).cuda()
    if a.sdpa!='auto':
        from torch.nn.attention import sdpa_kernel,SDPBackend
        backend=sdpa_kernel(SDPBackend.CUDNN_ATTENTION if a.sdpa=='cudnn' else SDPBackend.FLASH_ATTENTION)
    else: backend=contextlib.nullcontext()
    amp=lambda:torch.autocast('cuda',dtype=torch.float16) if a.mode=='amp' else contextlib.nullcontext()
    fn=lambda:demix(model,x) if a.full else model(x)
    with torch.inference_mode(),backend,amp():
        y=fn();torch.cuda.synchronize();init_ms=(time.perf_counter()-init)*1000
        for _ in range(a.warmup): y=fn()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(a.iters)]
        start=time.perf_counter()
        for begin,end in events: begin.record();y=fn();end.record()
        torch.cuda.synchronize();host=(time.perf_counter()-start)*1000
        times=[begin.elapsed_time(end) for begin,end in events]
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        out=y.float().cpu().numpy()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn();torch.cuda.synchronize()
        ops=[e.key for e in prof.key_averages() if '_scaled_dot_product' in e.key]
    from benchmark_matrix import metrics
    result={'schema_version':1,'implementation':'pytorch','pipeline':'waveform','mode':a.mode,'requested_sdpa':a.sdpa,'actual_sdpa_ops':ops,'precision':{'tf32':False,'input_resident_gpu':True,'output_resident_gpu':True},'samples':x.shape[-1],'shape':list(out.shape),'finite':bool(np.isfinite(out).all()),'timing_ms':{'initialization':init_ms,'warmup':a.warmup,'host_total':host,'host_per_iteration':host/a.iters,'cuda_event':times},'memory_bytes':{'allocated_peak':peak,'reserved_peak':reserved},'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}}
    if not a.full: result['correctness']=metrics(out,data['out'].reshape(out.shape))
    if not result['finite']: raise RuntimeError('nonfinite model output')
    if a.json: a.json.parent.mkdir(parents=True,exist_ok=True);a.json.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__': main()
