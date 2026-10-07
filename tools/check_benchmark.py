#!/usr/bin/env python3
"""Check waveform JSON, optionally enforcing same-fixture per-stem B0 gates."""
import argparse,json,math
from pathlib import Path

def check(value,baseline=None,min_snr=None):
    assert value['schema_version']==1,'unsupported benchmark schema'
    assert value['pipeline']=='waveform','CI requires the complete waveform boundary'
    shape=value['shape'];assert shape['batch']==1 and shape['channels']==2 and shape['stems']==6
    assert shape['samples']>0 and shape['sample_rate']==44100
    c=value['correctness'];assert c['finite'] and c['tail_covered']
    assert c['shape']==[1,6,2,shape['samples']]
    assert len(c['stems'])==6
    for i,s in enumerate(c['stems']):
        assert s['finite'] and math.isfinite(s['max_abs_error']),f'stem {i}: nonfinite'
        if s['reference_rms']<1e-8:assert s['max_abs_error']<=1e-6,f'stem {i}: silent-reference error'
        else:assert s['snr_db'] is not None and math.isfinite(s['snr_db']),f'stem {i}: invalid SNR'
    if min_snr is not None:assert c['overall']['snr_db']>=min_snr,'short golden SNR below gate'
    if baseline:
        for key in ['model_sha256','config_sha256','reference_sha256']:
            assert value['identity'][key]==baseline['identity'][key],f'{key} differs from B0'
        assert value['shape']==baseline['shape'],'B0 shape mismatch'
        for i,(a,b) in enumerate(zip(c['stems'],baseline['correctness']['stems'])):
            if b['reference_rms']>=1e-8:
                assert a['snr_db']>=b['snr_db']-0.1,f'stem {i}: SNR regression'
                assert a['max_abs_error']<=max(1.1*b['max_abs_error'],1e-6),f'stem {i}: absolute-error regression'
    return c['overall']['snr_db']

def main():
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--min-snr',type=float)
    a=p.parse_args();r=json.loads(a.result.read_text());b=json.loads(a.baseline.read_text())if a.baseline else None
    print('BENCHMARK_GATE_OK',check(r,b,a.min_snr))
if __name__=='__main__':main()
