import unittest
from unittest.mock import patch
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool


class WindowSamplerTests(unittest.TestCase):
    def test_pool_rejects_selection_dev_test_videos(self):
        base=AdaptiveBeam().base
        for video in [18,3,17]:
            with self.assertRaises(ValueError):TrainingPrefixPool(base,[video])

    def test_prefix_boundary_and_determinism(self):
        p=TrainingPrefixPool(AdaptiveBeam().base,[12],steps=100)
        for mode in ['uniform','stratified']:
            a=p.sample(91,mode);b=p.sample(91,mode)
            for k in a:np.testing.assert_array_equal(a[k],b[k])
            self.assertEqual(a['history'].shape,(4,32,2))
            self.assertEqual(a['truth'].shape,(4,100,2))
            self.assertTrue((a['start']>=63).all())
            self.assertTrue((a['start']+100<len(p.pool[12]['y'])).all())
            for h,y,t in zip(a['history'],a['truth'],a['start']):
                np.testing.assert_array_equal(h,p.pool[12]['y'][t-31:t+1])
                np.testing.assert_array_equal(y,p.pool[12]['y'][t+1:t+101])

    def test_changing_tail_cannot_change_pool_or_sampling(self):
        t=np.arange(800.)*.03;y=np.c_[np.sin(t),np.cos(t)]
        changed=y.copy();changed[400:]=1e5
        base=AdaptiveBeam().base
        with patch('training_window_sampler.load_video',return_value=y):a=TrainingPrefixPool(base,[12])
        with patch('training_window_sampler.load_video',return_value=changed):b=TrainingPrefixPool(base,[12])
        for mode in ['uniform','stratified']:
            sa=a.sample(72,mode);sb=b.sample(72,mode)
            for k in sa:np.testing.assert_array_equal(sa[k],sb[k])
        self.assertEqual(a.audit(),b.audit())

    def test_stratification_samples_cells_not_their_population_sizes(self):
        p=object.__new__(TrainingPrefixPool);p.videos=[12];p.steps=100
        p.seen=set();p.sampled_q=[];p.sampled_motion=[]
        p.pool={12:dict(y=np.zeros((300,2)),starts=np.arange(63,163),
                       q=np.r_[0,np.ones(99,dtype=int)],motion=np.ones(100),
                       cells=[np.array([0]),np.arange(1,100)])}
        for mode in ['uniform','stratified']:
            starts=[int(p.sample(seed,mode,per_video=1)['start'][0]) for seed in range(200)]
            fraction=np.mean(np.array(starts)==63)
            if mode=='uniform':self.assertLess(fraction,.08)
            else:self.assertTrue(.4<fraction<.6)


if __name__=='__main__':unittest.main()
