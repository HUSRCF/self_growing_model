import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from stateless_policy_adapter import StatelessPolicyAdapter
from feedback_distribution_pilot import inputs,corrected_tr
from evaluate_adapted_checked import checked_prefix,verify_checks
from v20_rnn_mixture.engine.data import tail_windows


MANIFEST=Path('adaptive_search_results/windows_uniform_seed1901_search_adapter.json')


class AdapterTests(unittest.TestCase):
    def test_read_matches_trained_controller_and_is_pure(self):
        s=AdaptiveBeam();wrapped=StatelessPolicyAdapter(s.machine,MANIFEST)
        h=tail_windows('dev',300,1)['history'];q,m=s.machine.initialize(h)
        before=m['hidden'].copy();pe,tr,read=s.machine.read(h,q,m)
        x=inputs(s,h,q,m['hidden'],pe,tr)
        expected=corrected_tr(wrapped.a,x,np.zeros((len(h),7)),tr,False)
        p,t,r=wrapped.read(h,q,m)
        np.testing.assert_array_equal(p,pe);np.testing.assert_array_equal(r['read_hidden'],read['read_hidden'])
        np.testing.assert_allclose(t,expected,rtol=0,atol=1e-14)
        np.testing.assert_array_equal(m['hidden'],before)
        p2,t2,r2=wrapped.read(h,q,m)
        np.testing.assert_array_equal(t,t2)
        np.testing.assert_allclose(t.sum(-1),1.)
        # Equal r creates equal commit memory, regardless of chosen e.
        a=wrapped.commit(m,q,q,np.zeros(len(q),int),r)
        b=wrapped.commit(m,q,q,np.full(len(q),7),r)
        np.testing.assert_array_equal(a['hidden'],b['hidden'])

    def test_manifest_rejects_feedback_contract_and_wrong_backbone(self):
        s=AdaptiveBeam();meta=json.loads(MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';meta['type']='event_feedback';p.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):StatelessPolicyAdapter(s.machine,p)
        s.machine.head.a['log_pi']=s.machine.head.a['log_pi'].copy()+.01
        with self.assertRaises(ValueError):StatelessPolicyAdapter(s.machine,MANIFEST)

    def test_missing_right_neighbor_invalidates_previous_output(self):
        p=np.arange(10.).reshape(1,1,5,2);h=np.zeros((1,32,2))
        failed=np.zeros((1,1,5),bool);failed[:,:,3:]=True
        out,bad,right=checked_prefix(p,failed,h,4)
        np.testing.assert_array_equal(bad,[[[False,False,True,True]]])
        np.testing.assert_array_equal(right,p[:,:,2])
        np.testing.assert_array_equal(out[:,:,2],out[:,:,1])
        np.testing.assert_array_equal(out[:,:,3],out[:,:,1])
        with self.assertRaises(ValueError):checked_prefix(p[:,:,:4],failed[:,:,:4],h,4)

    def test_model_hash_is_verified(self):
        meta=json.loads(MANIFEST.read_text())
        meta['model']=str((MANIFEST.parent/meta['model']).resolve());meta['model_sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json';p.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'model hash mismatch'):
                StatelessPolicyAdapter(AdaptiveBeam().machine,p)

    def test_verification_includes_terminal_point(self):
        class Checker:
            threshold=np.zeros(8)
            def score(self,left,middle,right,qs):return right[:,0]-10
        h=np.zeros((1,32,2));pred=np.zeros((1,1,3,2));q=np.zeros((1,1,3),int);bad=np.zeros((1,1,3),bool)
        result=verify_checks(Checker(),h,pred,np.zeros((1,1,2)),q,bad)
        self.assertEqual(result['checked_points'],3)
        with self.assertRaises(AssertionError):verify_checks(Checker(),h,pred,np.full((1,1,2),11.),q,bad)


if __name__=='__main__':unittest.main()
