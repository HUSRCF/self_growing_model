"""Read-only feature bridge for auditing the legacy internal-velocity writer."""
from pathlib import Path
import sys
import numpy as np
from v20_rnn_mixture.engine.dynamics import mlp_features


class LegacyFeatureBridge:
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def neural_values(self,h):
        b=self.base;d=b.local['shared'];a=(mlp_features(h,d['feature'])-b.xmean)/b.xscale
        for w,bias in zip(b.weights[:-1],b.bias[:-1]):
            a=a@w+bias;a=np.tanh(a) if d['activation']=='tanh' else np.maximum(a,0)
        return a@b.weights[-1]+b.bias[-1],np.c_[np.ones(len(h)),(a-b.hm)/b.hs]


def legacy_types():
    root=Path(__file__).resolve().parent/'v20_complete'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from v17.runtime import LocalMachine
    from v19.writers import Writer
    # Fail instead of silently importing unrelated same-named legacy packages.
    for name in ['v17.runtime','v19.writers']:
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(root)
    return LocalMachine,Writer
