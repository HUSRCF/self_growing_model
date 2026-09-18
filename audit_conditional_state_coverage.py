"""Frozen critic across predetermined actual states, matched reference/future RNG."""
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from adaptive_search_prototype import AdaptiveBeam
from training_window_sampler import TrainingPrefixPool
from audit_crossfit_value import continuation
from audit_energy_action_value import embedding
from audit_conditional_influence import worker
from audit_delayed_state import correlation
from crossfit_conditional_influence import state_features
from evaluate_conditional_policy import predict_choices


def summarize(rows,prior,choices,scores,video):
    y=np.asarray([r['labels'] for r in rows]);base=np.einsum('snrh,nr->snh',y,prior)
    idx=np.arange(len(prior));result={}
    for name,choice in choices.items():
        d=y[:,idx,choice]-base
        result[name]=dict(delta=float(d.mean()),seed_delta=d.mean((1,2)).tolist(),
                          horizon_delta=d.mean((0,1)).tolist(),seed_horizon_delta=d.mean(1).tolist(),
                          per_video_delta={str(v):float(d[:,video==v].mean()) for v in np.unique(video)},
                          choice_counts=np.bincount(choice,minlength=prior.shape[1]).tolist())
        if name in scores:
            centered=y.mean((0,3));centered-=centered.mean(1,keepdims=True)
            pred=scores[name];gap=pred[idx,choice]-(pred*prior).sum(1)
            result[name].update(centered_correlation=correlation(pred.ravel(),centered.ravel()),
                                predicted_delta=float(gap.mean()))
    return result


def main():
    directory=Path('adaptive_search_results')
    paths=[directory/'conditional_influence_critic_model.json',directory/'conditional_influence_critic_evaluation.json']
    model,old=[json.loads(p.read_text()) for p in paths]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,h in old['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;hashes[p]=h
    s=AdaptiveBeam();w=TrainingPrefixPool(s.base,steps=300).sample(401017,per_video=8)
    for k in ['video','start']:np.testing.assert_array_equal(w[k],model[k])
    snap={50:None};continuation(s,w['history'],761017,particles=8,snapshots=snap)
    states=[{k:snap[50][k][j::8].copy() for k in ['history','q','hidden','failed']} for j in range(4)]
    groups=[]
    for j,state in enumerate(states):
        assert not state['failed'].any()
        x,prior=state_features(s,state);choices,scores=predict_choices(model,x,w['video'])
        if j==0:
            for name in scores:
                np.testing.assert_array_equal(scores[name],model['scores'][name])
                np.testing.assert_array_equal(choices[name],model['choices'][name])
        groups.append(dict(particle_index=j,prior=prior.tolist(),choices={k:v.tolist() for k,v in choices.items()},
                           scores={k:v.tolist() for k,v in scores.items()},
                           state_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()},
                           q_counts=np.bincount(state['q'],minlength=8).tolist(),
                           q_mismatch_vs_first=float(np.mean(state['q']!=states[0]['q'])),
                           history_wrapped_rms_vs_first=float(np.sqrt(np.mean(np.angle(np.exp(1j*(state['history']-states[0]['history'])))**2))),
                           hidden_rms_vs_first=float(np.sqrt(np.mean((state['hidden']-states[0]['hidden'])**2)))))
    # Every state's policy is determined before generating any new labels.
    ref,rf=continuation(s,w['history'],811000,particles=16)
    reference=embedding(ref[:,:,[99,299]]);truth=embedding(w['truth'][:,[99,299]])
    replay=worker((states[0],truth,reference,811017))
    assert replay['seed']==old['rows'][0]['seed'];assert replay['guards']==old['rows'][0]['guards']
    np.testing.assert_array_equal(replay['labels'],old['rows'][0]['labels'])
    groups[0]['rows']=old['rows']
    print('First-state811017 labels exact; reuse all four old first-state rows',flush=True)
    tasks=[(j,seed) for j in [1,2,3] for seed in range(811017,811021)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (j,seed),row in zip(tasks,pool.map(worker,[(states[j],truth,reference,seed) for j,seed in tasks])):
            groups[j].setdefault('rows',[]).append(row)
            print('State',j,'future',seed,'guards',sum(row['guards']),flush=True)
    for group in groups:
        group['summary']=summarize(group['rows'],np.asarray(group['prior']),
                                   {k:np.asarray(v) for k,v in group['choices'].items()},
                                   {k:np.asarray(v) for k,v in group['scores'].items()},w['video'])
    for name,value in groups[0]['summary'].items():
        for key in ['delta','seed_delta','horizon_delta','per_video_delta']:
            assert value[key]==old['summary'][name][key]
    aggregate={name:dict(delta=float(np.mean([g['summary'][name]['delta'] for g in groups[1:]])),
                         horizon_delta=np.mean([g['summary'][name]['horizon_delta'] for g in groups[1:]],0).tolist())
               for name in groups[0]['summary']}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report=dict(groups=groups,new_states_aggregate=aggregate,source_hashes=hashes,reference_guards=int(rf.any(-1).sum()),
                video=w['video'].tolist(),start=w['start'].tolist(),
                note='TRAIN80 initial windows, fixed761017P8 pre-read50 particles0/1/2/3, no outcome-based selection. FrozenLOVOcritics, fixed811000P16reference and matched811017-20/P4per8roots future streams. Particle0 first stream exact replay,other3 oldrows reused; all oldsummary fields replayexact. Only intermediate states change. Q first-variation labels,NOT finite mixture gain. Sharedfuture/reference/window,not independent groups or newvideos; no refit/threshold/late/hold/DEV/TEST/defaultchange.')
    (directory/'conditional_state_coverage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(groups=[dict(particle_index=g['particle_index'],summary=g['summary'],q_mismatch=g['q_mismatch_vs_first']) for g in groups],new_states_aggregate=aggregate),indent=2),flush=True)


if __name__=='__main__':main()
