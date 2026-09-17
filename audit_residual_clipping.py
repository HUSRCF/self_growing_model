"""Read-only audit of unconstrained ridge fit versus clipped deployment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from continuous_residual_pilot import collect, design, correction
from drift_residual_pilot import generated_rows
from increment_residual_pilot import transport_targets
from v20_rnn_mixture.engine.common import SPLITS
from v20_rnn_mixture.engine.data import continuous_features


def clipping_scores(centers,truth,p,raw,cap):
    bounded=np.clip(raw,-cap,cap)
    target=np.angle(np.exp(1j*(truth[:,None]-centers)))
    emb=np.concatenate([np.sin(centers),np.cos(centers)],-1)
    truth_emb=np.concatenate([np.sin(truth),np.cos(truth)],-1)
    def weighted(value):return (p*value).sum(1)
    result=dict(saturated_candidate_probability=weighted((np.abs(raw)>cap).any(-1)),
                saturated_coordinate_fraction=weighted((np.abs(raw)>cap).mean(-1)))
    for name,d in [('zero',np.zeros_like(raw)),('raw',raw),('bounded',bounded)]:
        pred=np.concatenate([np.sin(centers+d),np.cos(centers+d)],-1)
        tangent=np.concatenate([np.cos(centers)*d,-np.sin(centers)*d],-1)
        result[name+'_embedding_mse']=weighted(((pred-truth_emb[:,None])**2).mean(-1))
        # Exactly the data term of fitted wrapped-angle quadratic, no rewrap after subtraction.
        result[name+'_fit_quadratic']=weighted(((target-d)**2).mean(-1))
        result[name+'_first_order_gain']=weighted((2*(truth_emb[:,None]-emb)*tangent).mean(-1))
        result[name+'_mean_square_correction']=weighted((d*d).mean(-1))
    return result


def main():
    root=Path('adaptive_search_results');s=AdaptiveBeam();observed=collect(s)
    generated=generated_rows(s,observed);transported=transport_targets(observed,generated)
    models={};hashes={};paths={}
    for name,filename in [('observed','drift_residual_model_observed.npz'),
                          ('absolute','increment_residual_model_absolute.npz'),
                          ('increment','increment_residual_model_increment.npz')]:
        path=root/filename;paths[name]=path;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:models[name]={k:z[k].copy() for k in z.files}
    report={}
    for carrier,data in [('observed',observed),('generated',generated)]:
        h=data['history'];q=data['q'];n=len(q);p=data['probability']
        hh=np.repeat(h,8,axis=0);qq=np.repeat(q,8);rr=np.tile(np.arange(8),n)
        centers=s.base.execute_rule(hh,qq,rr).reshape(n,8,2)
        features=np.repeat(continuous_features(h,s.base),8,axis=0)
        report[carrier]={}
        for name,model in models.items():
            x=design(features,qq,rr,model['mean'],model['scale'])
            raw=(x@model['coef']).reshape(n,8,2)
            np.testing.assert_array_equal(np.clip(raw,-model['cap'],model['cap']),
                correction(model,s.base,hh,qq,rr).reshape(n,8,2))
            report[carrier][name]={}
            targets=[('absolute',data['truth'])]
            if carrier=='generated':targets.append(('increment',transported['truth']))
            for label,truth in targets:
                rows=clipping_scores(centers,truth,p,raw,model['cap']);parts={}
                for split,videos in [('fit',SPLITS['train'][:-3]),('hold',SPLITS['train'][-3:])]:
                    mask=np.isin(data['video'],videos)
                    parts[split]={k:float(v[mask].mean()) for k,v in rows.items()}
                    parts[split]['per_video']={str(video):{k:float(v[data['video']==video].mean())
                        for k,v in rows.items()} for video in videos}
                report[carrier][name][label]=parts
        print(carrier,'done',flush=True)
    for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[name]
    output=dict(groups=report,model_sha256=hashes,models_unchanged=True,deployed_output_exact=True,
                note='TRAIN prefixes only; fixed models, no strength selection/refit/DEV/TEST. Raw outputs evaluated only at fixed states, never deployed. Positive first-order gain helpful. Quadratic is original wrapped residual minus correction squared, averaged over angles; embedding MSE uses four sin/cos coordinates. Saturation is weighted by current q-destination law. No causal claim about full-rollout outcome from local scores.')
    (root/'residual_clipping_audit.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
