"""Common output-angle shifts preserve ensemble pair distances on the torus."""
import numpy as np
from scipy.optimize import minimize
from audit_energy_action_value import embedding
from ensemble_score_objective import energy_costs


def attraction_gradient(points,truth,shift):
    shifted=points+np.asarray(shift)
    distance=np.linalg.norm(embedding(shifted)-embedding(truth)[:,None],axis=-1)
    numerator=np.sin(shifted-truth[:,None])
    derivative=np.divide(numerator,distance[...,None],out=np.zeros_like(numerator),where=distance[...,None]>0)
    return distance.mean(1),derivative.mean(1)


def shifted_cost(points,truth,failed,shift):
    baseline=energy_costs(embedding(points),embedding(truth),failed)[0]
    a,_=attraction_gradient(points,truth,shift);zero,_=attraction_gradient(points,truth,np.zeros(2))
    return baseline+(a-zero)


def fit_shift(points,truth):
    zero,_=attraction_gradient(points,truth,np.zeros(2))
    def objective(shift):
        a,g=attraction_gradient(points,truth,shift)
        return float((a-zero).mean()+.001*np.sum(shift**2)),g.mean(0)+.002*shift
    result=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-.25,.25)]*2,
                    options=dict(maxiter=200,gtol=1e-9,ftol=1e-12))
    use_shift=bool(result.success and result.fun<0)
    return dict(shift=result.x.tolist() if use_shift else [0.,0.],candidate_shift=result.x.tolist(),
                success=bool(result.success),use_shift=use_shift,message=str(result.message),iterations=int(result.nit),
                penalized_delta=float(result.fun),at_boundary=(np.abs(result.x)>=.25-1e-8).tolist(),
                note='Output only,perhorizon common2angle shift,bounds±.25rad,L2.001,zero start,TRAIN zero fallback.')
