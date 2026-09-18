"""Two-region risk on improvement over the exact point baseline, not absolute loss."""
import numpy as np
from scipy.optimize import minimize
from soft_kernel_gate import scalar_optimum


def group_coefficients(linear,quadratic,groups):
    labels=np.unique(groups)
    return np.array([np.mean(linear[groups==g]) for g in labels]),np.array([np.mean(quadratic[groups==g]) for g in labels])


def worst_scalar(linear,quadratic,groups):
    l,q=group_coefficients(np.asarray(linear),np.asarray(quadratic),np.asarray(groups))
    candidates=[0.,1.]
    for a,b in zip(l,q):
        if b>0:candidates.append(float(np.clip(-a/(2*b),0,1)))
    # All curves pass through zero; the other pairwise crossing is analytic.
    for i in range(len(l)):
        for j in range(i):
            if q[i]!=q[j]:
                a=-(l[i]-l[j])/(q[i]-q[j])
                if 0<a<1:candidates.append(float(a))
    return min(candidates,key=lambda a:np.max(l*a+q*a*a))


def risks(beta,z,linear,quadratic,groups,mode):
    raw=beta[0]+beta[1]*z;a=np.clip(raw,0,1)
    loss=linear*a+quadratic*a*a
    d=(linear+2*quadratic*a)*((raw>0)&(raw<1))
    sample_grad=np.c_[d,d*z]
    if mode=='mean':return np.array([loss.mean()]),sample_grad.mean(0)[None]
    if mode!='worst':raise ValueError('Unknown risk mode')
    masks=[groups==g for g in np.unique(groups)]
    return np.array([loss[m].mean() for m in masks]),np.array([sample_grad[m].mean(0) for m in masks])


def fit_gate(log_motion,linear,quadratic,groups,mode='worst'):
    x,l,q=np.asarray(log_motion,float),np.asarray(linear,float),np.asarray(quadratic,float)
    groups=np.asarray(groups)
    if x.ndim!=1 or not len(x) or any(a.shape!=x.shape for a in [l,q,groups]) or not all(np.isfinite(a).all() for a in [x,l,q]):
        raise ValueError('Matching finite nonempty vectors required')
    mean,scale=float(x.mean()),max(float(x.std()),1e-6);z=np.clip((x-mean)/scale,-3,3)
    scalar=worst_scalar(l,q,groups) if mode=='worst' else scalar_optimum(l,q)
    scalar_loss=float(risks([scalar,0],z,l,q,groups,mode)[0].max())
    initial=np.array([.5,0.,float(risks([.5,0],z,l,q,groups,mode)[0].max())])
    def objective(v):return v[2]+.001*v[1]**2,np.array([0.,.002*v[1],1.])
    def constraint(v):return v[2]-risks(v[:2],z,l,q,groups,mode)[0]
    def jacobian(v):return np.c_[-risks(v[:2],z,l,q,groups,mode)[1],np.ones(len(constraint(v)))]
    result=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(-2,3),(-2,2),(None,None)],
                    constraints=[dict(type='ineq',fun=constraint,jac=jacobian)],options=dict(maxiter=200,ftol=1e-12))
    actual=float(risks(result.x[:2],z,l,q,groups,mode)[0].max()+.001*result.x[1]**2)
    violation=float(max(0.,-constraint(result.x).min()))
    selected=bool(result.success and violation<=1e-8 and actual<scalar_loss)
    return dict(mean=mean,scale=scale,scalar=float(scalar),scalar_loss=scalar_loss,beta=result.x[:2].tolist(),
                success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                use_affine=selected,candidate_loss=actual,constraint_violation=violation,mode=mode,
                note='SLSQP epigraph,one[.5,0]start,bounds[-2,3]/[-2,2],.001slopeL2;exact scalar fallback. Local motion optimum only.')
