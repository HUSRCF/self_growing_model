"""Manifest-verified q-logit adapter with NO event-specific feedback state."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from v20_rnn_mixture.engine.event import EventMachine
from v20_rnn_mixture.engine.dynamics import Dynamics
from v20_rnn_mixture.engine.data import continuous_features


def fingerprint(arrays):
    h=hashlib.sha256()
    for k,v in sorted(arrays.items()):
        a=np.asarray(v)
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def export_manifest(summary_path):
    summary_path=Path(summary_path);summary=json.loads(summary_path.read_text())
    if summary['config'].get('window_sampling')!='uniform' or 'closed_loop' not in summary['training']:
        raise ValueError('Expected uniform-refresh no-feedback training report')
    model=summary_path.with_name(summary_path.stem+'_closed_loop.npz')
    machine=EventMachine(Dynamics())
    manifest=dict(type='q_logit_residual_no_feedback_v1',model=model.name,
                  model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  head_fingerprint=fingerprint(machine.head.a),
                  training_summary=summary_path.name,training_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                  training_arm='closed_loop',selected_epoch=summary['training']['closed_loop']['selected_epoch'])
    out=summary_path.with_name(summary_path.stem+'_search_adapter.json')
    out.write_text(json.dumps(manifest,indent=2));return out


class StatelessPolicyAdapter:
    def __init__(self,machine,manifest_path):
        if not isinstance(machine,EventMachine) or machine.reset:
            raise ValueError('Adapter requires the non-reset frozen EventMachine')
        path=Path(manifest_path);m=json.loads(path.read_text())
        if m.get('type')!='q_logit_residual_no_feedback_v1' or m.get('training_arm')!='closed_loop':
            raise ValueError('Destination-only bans require a no-feedback adapter contract')
        source=path.parent/m['model']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=m['model_sha256']:
            raise ValueError('Adapter model hash mismatch')
        if fingerprint(machine.head.a)!=m['head_fingerprint']:
            raise ValueError('Frozen backbone mismatch')
        with np.load(source,allow_pickle=False) as z:self.a={k:z[k].copy() for k in z.files}
        self.inner=machine;self.base=machine.base;self.head=machine.head;self.k=machine.k
        d=len(self.head.a['mean'])+self.k+self.head.width+self.k+7;a=self.a
        if a['mean'].shape!=(d,) or a['scale'].shape!=(d,) or a['w1'].ndim!=2 or a['w1'].shape[0]!=d or a['w2'].shape!=(a['w1'].shape[1],self.k) or a['b1'].shape!=(a['w1'].shape[1],) or a['b2'].shape!=(self.k,):
            raise ValueError('Adapter feature dimensions mismatch')
        if not all(np.isfinite(v).all() for v in a.values()) or (a['scale']<=0).any():
            raise ValueError('Nonfinite/invalid adapter parameters')

    def initialize(self,history):return self.inner.initialize(history)

    def read(self,h,q,memory):
        raw=continuous_features(h,self.base)
        pe,tr,nxt=self.head.step(raw,q,memory['hidden'],reset=False)
        marginal=np.einsum('be,ber->br',pe,tr)
        x=np.c_[self.head.transform(raw),np.eye(self.k)[q],memory['hidden'],
                np.log(np.maximum(marginal,1e-12)),np.zeros((len(h),7))]
        a=self.a;z=np.clip((x-a['mean'])/a['scale'],-8,8)
        correction=np.tanh(z@a['w1']+a['b1'])@a['w2']+a['b2']
        return pe,softmax(np.log(np.maximum(tr,1e-12))+correction[:,None,:],axis=-1),{'read_hidden':nxt}

    def commit(self,memory,q,r,e,read_result):return self.inner.commit(memory,q,r,e,read_result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary');a=p.parse_args();print(export_manifest(a.summary))
