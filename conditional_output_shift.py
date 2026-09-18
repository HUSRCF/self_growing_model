"""Small history-motion-conditioned output shift, constant within each ensemble."""
import numpy as np
from scipy.optimize import minimize
from output_center_shift import attraction_gradient


def objective(flat,x,points,truth,zero):
    beta=flat.reshape(2,2);raw=x@beta;shift=np.clip(raw,-.25,.25)
    a,g=attraction_gradient(points,truth,shift[:,None,:])
    g*=((raw>-.25)&(raw<.25))
    return float((a-zero).mean()+.001*np.sum(beta**2)),(x.T@g/len(x)+.002*beta).ravel()


def fit_model(log_motion,points,truth):
    mean=float(log_motion.mean());scale=max(float(log_motion.std()),1e-6)
    x=np.c_[np.ones(len(log_motion)),np.clip((log_motion-mean)/scale,-3,3)]
    zero=attraction_gradient(points,truth,[0.,0.])[0]
    result=minimize(objective,np.zeros(4),args=(x,points,truth,zero),jac=True,method='L-BFGS-B',
                    bounds=[(-.25,.25)]*4,options=dict(maxiter=200,gtol=1e-9,ftol=1e-12))
    return dict(mean=mean,scale=scale,beta=result.x.reshape(2,2).tolist(),success=bool(result.success),
                message=str(result.message),iterations=int(result.nit),penalized_delta=float(result.fun),
                use_shift=bool(result.success and result.fun<0),at_boundary=(np.abs(result.x)>=.25-1e-8).tolist())


def predict(model,log_motion):
    if not model['use_shift']:return np.zeros((len(log_motion),2))
    x=np.c_[np.ones(len(log_motion)),np.clip((log_motion-model['mean'])/model['scale'],-3,3)]
    return np.clip(x@np.asarray(model['beta']),-.25,.25)
