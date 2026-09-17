"""Energy U-statistic and causal leave-one-trajectory-out baselines."""
import numpy as np


def energy_costs(prediction,truth,failed):
    """Return one cost/window and baselines/window/particle, excluding self.

    prediction: [windows, particles, embedding dimensions]. All pair terms
    incident to the omitted particle must be removed from its baseline.
    Multiply cost-minus-baseline by P when the REINFORCE loss averages P
    log probabilities. This recovers the gradient of the ensemble objective,
    not a gradient attenuated by the number of particles.
    """
    x=np.asarray(prediction);y=np.asarray(truth);dead=np.asarray(failed,dtype=float)
    if x.ndim!=3 or y.shape!=(x.shape[0],x.shape[2]) or dead.shape!=x.shape[:2]:
        raise ValueError('Expected prediction[N,P,D],truth[N,D],failed[N,P]')
    p=x.shape[1]
    if p<3:raise ValueError('Energy leave-out control requires at least three particles')
    distance=np.linalg.norm(x-y[:,None],axis=-1)
    pair=np.linalg.norm(x[:,:,None]-x[:,None,:],axis=-1)
    total=pair.sum((1,2));row=pair.sum(2)
    cost=distance.mean(1)-total/(2*p*(p-1))+2*dead.mean(1)
    baseline=(distance.sum(1,keepdims=True)-distance)/(p-1)
    baseline-=(total[:,None]-2*row)/(2*(p-1)*(p-2))
    baseline+=2*(dead.sum(1,keepdims=True)-dead)/(p-1)
    return cost,baseline
