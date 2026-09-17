"""Same frozen F parameters, history-only work shared across destinations.

This is a numerical optimization, not a learned rule change. Matrix shape
changes can produce floating-point roundoff; compare to the frozen writer.
"""
import numpy as np
from v20_rnn_mixture.engine.dynamics import mlp_features
from v20_rnn_mixture.engine.features import rule_features


class SharedCandidateWriter:
    def __init__(self,base):
        self.base=base
        degrees=np.array([len(t) for t in base.mat['terms']])
        self.levels=[np.flatnonzero(degrees==d) for d in range(1,int(degrees.max())+1)]
        self.parents=np.asarray(base.mat['parents'])
        self.last=np.asarray(base.mat['last'])

    def execute(self,h,q,rs):
        if len(h)!=1:raise ValueError('One shared parent history is required')
        b=self.base;d=b.local['shared']
        a=(mlp_features(h,d['feature'])-b.xmean)/b.xscale
        for w,bias in zip(b.weights[:-1],b.bias[:-1]):
            a=a@w+bias
            a=np.tanh(a) if d['activation']=='tanh' else np.maximum(a,0)
        global_std=a@b.weights[-1]+b.bias[-1]
        z=np.c_[np.ones(1),(a-b.hm)/b.hs]
        correction=np.einsum('bp,bpd->bd',np.repeat(z,len(rs),axis=0),b.delta[q,rs])
        acc=(global_std+correction)*b.yscale+b.ymean
        inertial=2*h[:,-1]-h[:,-2]
        neural=inertial+acc
        fraction=float(b.local.get('neural_fraction',1.))
        if fraction==1.:return neural
        features=rule_features(h,b.observer,b.rules.get('memory_lags',()))
        phi=np.ones((1,len(b.mat['terms'])))
        for ids in self.levels:
            phi[:,ids]=phi[:,self.parents[ids]]*features[:,self.last[ids]]
        polynomial=inertial+np.einsum('bp,bpd->bd',np.repeat(phi,len(rs),axis=0),b.coefficients[q,rs])
        return fraction*neural+(1-fraction)*polynomial
