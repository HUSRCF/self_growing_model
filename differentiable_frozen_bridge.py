"""CPU float64 frozen F/GRU bridge; no trainable backbone or guard policy."""
import numpy as np
import torch
from v20_rnn_mixture.engine.features import jet_stencil


def tensor(value):
    return torch.tensor(np.asarray(value).copy(),dtype=torch.float64,device='cpu')


class FrozenBridge:
    def __init__(self,s):
        b=s.base;self.k=b.k;self.reset=s.machine.reset;self.width=s.machine.head.width
        self.observer=b.observer;self.feature=b.local['shared']['feature'];self.activation=b.local['shared']['activation']
        self.fraction=float(b.local.get('neural_fraction',1.));self.lags=b.rules.get('memory_lags',())
        self.stencil=tensor(b.stencil);self.summary_stencil=tensor(jet_stencil())
        self.partition={k:tensor(b.partition[k]) for k in ['center','scale','weights','centers']}
        self.order=b.partition['order'];self.weights=[tensor(v) for v in b.weights];self.bias=[tensor(v) for v in b.bias]
        for name in ['xmean','xscale','ymean','yscale','hm','hs','delta','coefficients']:setattr(self,name,tensor(getattr(b,name)))
        degree=b.mat['degree']
        self.terms=torch.tensor([[j+1 for j in term]+[0]*(degree-len(term)) for term in b.mat['terms']],dtype=torch.long)
        self.head={k:tensor(v) for k,v in s.machine.head.a.items()}

    def jets(self,h,stencil):
        recent=h[:,-stencil.shape[1]:]
        return torch.einsum('kw,bwd->bkd',stencil,recent-recent[:,-1:])

    def read_features(self,h,observer='angles'):
        y=h[:,-1];d=(h[:,-1]-h[:,-2])/.2
        return torch.cat([torch.sin(y),torch.cos(y),d],1) if observer=='angles' else torch.cat([y,d],1)

    def context(self,h):
        d=torch.diff(h,dim=1)
        return torch.cat([torch.log1p(((d/.2)**2).mean(1)),torch.std(d,dim=1,correction=0)/.2,d.mean(1)/.2],1)

    def partition_features(self,h):
        raw=self.jets(h,self.stencil)[:,:self.order].reshape(len(h),-1);p=self.partition
        return (raw-p['center'])/p['scale']*p['weights']

    def state(self,h):
        f=self.partition_features(h)
        return ((f[:,None]-self.partition['centers'][None])**2).sum(-1).argmin(1)

    def continuous_features(self,h):
        return torch.cat([self.read_features(h,self.observer),self.partition_features(h),self.context(h)],1)

    def mlp_features(self,h):
        if self.feature=='local':return self.read_features(h)
        if self.feature=='summary':return torch.cat([self.read_features(h),self.jets(h,self.summary_stencil).reshape(len(h),-1),self.context(h)],1)
        if self.feature=='full':return torch.cat([torch.sin(h).reshape(len(h),-1),torch.cos(h).reshape(len(h),-1),torch.diff(h,dim=1).reshape(len(h),-1)/.2],1)
        raise ValueError('unsupported feature')

    def execute(self,h,q,r):
        a=(self.mlp_features(h)-self.xmean)/self.xscale
        for w,b in zip(self.weights[:-1],self.bias[:-1]):
            a=a@w+b;a=torch.tanh(a) if self.activation=='tanh' else torch.relu(a)
        global_std=a@self.weights[-1]+self.bias[-1]
        z=torch.cat([torch.ones_like(a[:,:1]),(a-self.hm)/self.hs],1)
        correction=torch.einsum('bp,bpd->bd',z,self.delta[q,r])
        neural=2*h[:,-1]-h[:,-2]+(global_std+correction)*self.yscale+self.ymean
        if self.fraction==1:return neural
        z=self.read_features(h,self.observer)
        if self.lags:z=torch.cat([z]+[(h[:,-1]-h[:,-1-lag])/(.2*lag) for lag in self.lags],1)
        augmented=torch.cat([torch.ones_like(z[:,:1]),z],1)
        phi=augmented[:,self.terms].prod(-1)
        polynomial=2*h[:,-1]-h[:,-2]+torch.einsum('bp,bpd->bd',phi,self.coefficients[q,r])
        return self.fraction*neural+(1-self.fraction)*polynomial

    def read(self,h,q,hidden):
        a=self.head;z=torch.clamp((self.continuous_features(h)-a['mean'])/a['scale'],-8,8)
        emb=a['embedding.weight'][q];inp=torch.cat([z,emb],1)
        previous=torch.zeros_like(hidden) if self.reset else hidden
        gi=inp@a['gru.weight_ih_l0'].T+a['gru.bias_ih_l0'];gh=previous@a['gru.weight_hh_l0'].T+a['gru.bias_hh_l0']
        ir,iz,inn=gi.chunk(3,dim=-1);hr,hz,hn=gh.chunk(3,dim=-1)
        reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz)
        nxt=(1-update)*torch.tanh(inn+reset*hn)+update*previous
        read=torch.cat([z,emb,nxt],1)
        le=read@a['event.weight'].T+a['event.bias']+a['log_pi'][q]
        lt=(read@a['transition.weight'].T+a['transition.bias']).reshape(-1,int(a['events']),self.k)+a['log_T'][q]
        return torch.softmax(le,-1),torch.softmax(lt,-1),nxt

    def initialize(self,h):
        n,L,_=h.shape;hidden=torch.zeros((n,self.width),dtype=h.dtype,device=h.device)
        for end in range(max(self.stencil.shape[1],L-64),L):
            past=h[:,max(0,end-32):end]
            if past.shape[1]<32:past=torch.cat([past[:,:1].expand(-1,32-past.shape[1],-1),past],1)
            _,_,hidden=self.read(past,self.state(past),hidden)
        return self.state(h[:,-32:]),hidden


def fixed_path(bridge,h,events,destinations,theta):
    q,hidden=bridge.initialize(h);logp=torch.zeros(len(h),dtype=h.dtype);out=[]
    for e,r in zip(events,destinations):
        pe,T,hidden=bridge.read(h,q,hidden);i=torch.arange(len(h))
        logp=logp+torch.log(pe[i,e])+torch.log(T[i,e,r])
        y=bridge.execute(h,q,r)+theta
        out.append(y);h=torch.cat([h[:,1:],y[:,None]],1);q=r
    return torch.stack(out,1),logp


def sampled_path(bridge,h,steps,theta,seed,trace=False):
    """Match unchecked NumPy rollout, including continuing q/hidden after failure.

    Sampling is detached; returned logp retains routing-state derivatives.
    Hard failure decisions are not differentiable. This is not a proof of an
    unbiased gradient across parameter-dependent failure boundaries.
    """
    from feedback_distribution_pilot import sample
    q,hidden=bridge.initialize(h);rng=np.random.default_rng(seed)
    dead=torch.zeros(len(h),dtype=torch.bool);logp=torch.zeros(len(h),dtype=h.dtype)
    out=[];failures=[];records=[];prefix_logs=[];i=torch.arange(len(h));marginal_logp=torch.zeros_like(logp)
    for _ in range(steps):
        pe,T,hidden=bridge.read(h,q,hidden)
        e=torch.tensor(sample(pe.detach().numpy(),rng.random(len(h))))
        r=torch.tensor(sample(T[i,e].detach().numpy(),rng.random(len(h))))
        logp=logp+torch.log(pe[i,e])+torch.log(T[i,e,r])
        prefix_logs.append(logp)
        marginal=(pe[:,:,None]*T).sum(1)
        marginal_logp=marginal_logp+torch.log(marginal[i,r])
        candidate=bridge.execute(h,q,r)+theta
        dead=dead|(~torch.isfinite(candidate)).any(1)|((candidate-h[:,-1]).abs()>np.pi).any(1)
        y=torch.where(dead[:,None],h[:,-1],candidate)
        h=torch.cat([h[:,1:],y[:,None]],1);q=r
        out.append(y);failures.append(dead)
        if trace:records.append(dict(event=e.clone(),q=q.clone(),hidden=hidden.clone(),history=h.clone()))
    return dict(prediction=torch.stack(out,1),failed=torch.stack(failures,1),logp=logp,
                history=h,q=q,hidden=hidden,trace=records,prefix_logp=torch.stack(prefix_logs,1),marginal_logp=marginal_logp)
