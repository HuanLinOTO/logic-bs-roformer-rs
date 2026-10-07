"""Numerical gate and full-song boundaries; use the reference Python env."""
import unittest
import numpy as np
from benchmark_matrix import metrics,numerical_gate

class NumericalGateTests(unittest.TestCase):
    def test_silence_uses_absolute_error(self):
        ref=np.zeros((1,6,2,1025),np.float32);base=metrics(ref,ref)
        self.assertFalse(numerical_gate(base,metrics(ref+5e-7,ref)))
        self.assertTrue(numerical_gate(base,metrics(ref+2e-6,ref)))
    def test_single_stem_regression_is_not_hidden(self):
        ref=np.ones((1,6,2,1025),np.float32);base=ref+1e-4;cand=base.copy();cand[:,5]+=1e-3
        self.assertTrue(numerical_gate(metrics(base,ref),metrics(cand,ref)))
    def test_nonfinite_is_rejected(self):
        ref=np.ones((1,6,2,1025),np.float32);bad=ref.copy();bad[0,0,0,0]=np.nan
        self.assertTrue(numerical_gate(metrics(ref,ref),metrics(bad,ref)))

class ChunkTests(unittest.TestCase):
    def test_grid_covers_tail(self):
        from bench_ref import chunk_starts,CHUNK,STEP,BORDER
        for length in [1,37,44100*8,CHUNK,44100*30,STEP*3+91]:
            starts=chunk_starts(length)
            self.assertGreaterEqual(starts[-1]+CHUNK,BORDER+length)
            self.assertEqual(starts,list(range(0,starts[-1]+1,STEP)))
        self.assertEqual(len(chunk_starts(44100*30)),3)
        with self.assertRaises(ValueError): chunk_starts(0)
    def test_reflection_matches_numpy_for_multiple_periods(self):
        import torch
        from bench_ref import reflect_pad
        x=np.array([[1,2,3],[10,20,30]],dtype=np.float32)
        expected=np.pad(x,((0,0),(17,29)),mode='reflect')
        np.testing.assert_array_equal(reflect_pad(torch.from_numpy(x),17,29).numpy(),expected)
    def test_demix_shape_channels_and_last_impulse(self):
        import torch
        from bench_ref import demix,CHUNK
        class Identity:
            def __call__(self,x): return x[:,None].expand(-1,6,-1,-1)
        for length in [1,37,44100*8,CHUNK,44100*30,559360*3+91]:
            x=torch.zeros(1,2,length);x[0,0,-1]=1;x[0,1,0]=0.25
            result=demix(Identity(),x)
            self.assertEqual(tuple(result.shape),(1,6,2,length))
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertGreater(float(result[0,0,0,-1]),0)
            self.assertGreater(float(result[0,0,1,0]),0)
            self.assertEqual(float(result[0,0,1,-1]),0 if length>1 else 0.25)
if __name__=='__main__': unittest.main()
