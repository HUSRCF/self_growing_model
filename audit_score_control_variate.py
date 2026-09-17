"""Fit score controls on eight RNG seeds; evaluate eight disjoint RNG seeds."""
import json
from pathlib import Path
import numpy as np
import torch
from score_control_variate import fit_coefficient
from audit_source_q_gradient import stability
from adaptive_search_prototype import AdaptiveBeam
from differentiable_frozen_bridge import FrozenBridge,tensor,sampled_path
from training_window_sampler import TrainingPrefixPool
from train_hybrid_writer import hybrid_loss


def main():
    torch.set_num_threads(1);s=AdaptiveBeam();b=FrozenBridge(s);root=Path('adaptive_search_results')
    w=TrainingPrefixPool(s.base,steps=300).sample(190101,per_video=1)
    old=json.loads((root/'source_q_gradient_audit.json').read_text())
    np.testing.assert_array_equal(w['start'],old['window_start'])
    with np.load(root/'continuous_residual_model_ridge_0.0001.npz') as f:bound=tensor(f['cap']*.01)
    phases={};coefficients=None
    for phase,seeds in [('fit',range(371017,371025)),('evaluation',range(381017,381025))]:
        rows=[]
        for seed in seeds:
            gradients=[];scores=[];guards=0
            for j,h in enumerate(w['history']):
                theta=torch.zeros((8,2),dtype=torch.float64,requires_grad=True)
                a=sampled_path(b,tensor(np.repeat(h[None],4,axis=0)),300,theta*bound,seed+j*100000,detach_every=50)
                _,loss=hybrid_loss(a['prediction'],tensor(w['truth'][j]),a['failed'],a['logp'])
                g=torch.autograd.grad(loss,theta,retain_graph=True)[0].detach().numpy()
                control=torch.autograd.grad(a['logp'].sum(),theta)[0].detach().numpy()
                assert np.isfinite(g).all() and np.isfinite(control).all()
                if phase=='fit':
                    reference=next(r for r in old['rows'] if r['seed']==seed)['windows'][j]['gradient']
                    np.testing.assert_array_equal(g,reference)
                gradients.append(g.tolist());scores.append(control.tolist());guards+=int(a['failed'].any(1).sum())
                del a,loss
            rows.append(dict(seed=seed,gradients=gradients,score_gradients=scores,guard_particles=guards))
            print(phase,seed,'guards',guards,flush=True)
        phases[phase]=rows
        if phase=='fit':
            g=np.array([r['gradients'] for r in rows]);c=np.array([r['score_gradients'] for r in rows])
            coefficients=np.array([fit_coefficient(g[:,j].reshape(8,-1),c[:,j].reshape(8,-1)) for j in range(10)])
            print('frozen coefficients',coefficients.tolist(),flush=True)
    summary={}
    for phase,rows in phases.items():
        g=np.array([r['gradients'] for r in rows]);c=np.array([r['score_gradients'] for r in rows]);adjusted=g-coefficients[None,:,None,None]*c
        summary[phase]={'raw':stability(g.mean(1)),'adjusted':stability(adjusted.mean(1)),
                        'raw_shared':stability(g.mean(1).sum(1)),'adjusted_shared':stability(adjusted.mean(1).sum(1)),
                        'per_window_variance_ratio':[float(np.sum(adjusted[:,j].var(0,ddof=1))/np.sum(g[:,j].var(0,ddof=1))) for j in range(10)]}
    report=dict(phases=phases,coefficients=coefficients.tolist(),summary=summary,fit_gradient_exact_replay=True,video=w['video'].tolist(),window_start=w['start'].tolist(),
                note='Fit8seeds371017–24; independent evaluation8seeds381017–24,same fixedtheta0/10fitTRAINwindows/P4/300/truncation50. Scalar coefficient perwindow,16Dnorm fit,retained LOO; no fitting on evaluation samples. In-sample fit statistics optimistic and not unbiasedness evidence. Preserves only expectation of chosen truncated estimator under zero-mean score assumptions,not original full gradient. No training/hold/DEV/TEST.')
    (root/'score_control_variate_audit.json').write_text(json.dumps(report,indent=2))
    print({phase:{k:(v['variance_trace'],v['pairwise_cosine_mean']) for k,v in result.items() if k!='per_window_variance_ratio'} for phase,result in summary.items()},flush=True)


if __name__=='__main__':main()
