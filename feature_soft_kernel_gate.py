"""Fixed eight-feature compression of causal model state for a small soft gate."""
import numpy as np
from scipy.optimize import minimize
from soft_kernel_gate import scalar_optimum


def fit_mapping(raw):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite nonempty feature matrix required')
    d=raw.shape[1]
    projection=np.random.default_rng(1901).normal(size=(d,8))/np.sqrt(d)
    return dict(mean=raw.mean(0).tolist(),scale=np.maximum(raw.std(0),1e-5).tolist(),projection=projection.tolist())


def design(mapping,raw):
    raw=np.asarray(raw,float)
    if raw.ndim!=2 or raw.shape[1]!=len(mapping['mean']) or not np.isfinite(raw).all():
        raise ValueError('Matching finite causal feature matrix required')
    z=np.clip((raw-np.asarray(mapping['mean']))/np.asarray(mapping['scale']),-3,3)
    return np.c_[np.ones(len(raw)),np.tanh(z@np.asarray(mapping['projection']))]


def objective(beta,x,linear,quadratic,penalty=.001):
    raw=x@beta;alpha=np.clip(raw,0,1)
    derivative=(linear+2*quadratic*alpha)*((raw>0)&(raw<1))
    loss=np.mean(linear*alpha+quadratic*alpha**2)+penalty*np.sum(beta[1:]**2)
    gradient=x.T@derivative/len(x);gradient[1:]+=2*penalty*beta[1:]
    return float(loss),gradient


def fit_gate(x,linear,quadratic):
    x=np.asarray(x,float);linear=np.asarray(linear,float);quadratic=np.asarray(quadratic,float)
    scalar=scalar_optimum(linear,quadratic)
    scalar_loss=float(np.mean(linear*scalar+quadratic*scalar**2))
    initial=np.zeros(x.shape[1]);initial[0]=.5
    result=minimize(objective,initial,args=(x,linear,quadratic),jac=True,method='L-BFGS-B',
                    bounds=[(-2,3)]+[(-2,2)]*(len(initial)-1),options=dict(maxiter=200,gtol=1e-9,ftol=1e-12))
    return dict(beta=result.x.tolist(),scalar=scalar,scalar_loss=scalar_loss,candidate_loss=float(result.fun),
                use_features=bool(result.success and result.fun<scalar_loss),success=bool(result.success),
                message=str(result.message),iterations=int(result.nit))


def predict_gate(model,x):
    if not model['use_features']:return np.full(len(x),model['scalar'],float)
    return np.clip(np.asarray(x)@np.asarray(model['beta']),0,1)
