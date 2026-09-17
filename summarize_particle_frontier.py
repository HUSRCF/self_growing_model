"""Compute/quality frontier, with both empirical and off-diagonal energy."""
import json
from pathlib import Path
import numpy as np
from v20_rnn_mixture.engine.data import tail_windows


def energy_u(empirical,mean_distance,particles):
    """Remove self-pair finite-P bias; assumes exchangeable IID samples for unbiased population interpretation."""
    if particles<2:return None
    return (particles*empirical-mean_distance)/(particles-1)


def aggregate(runs):
    return {str(t):{k:float(np.mean([r['score'][str(t)][k] for r in runs]))
                   for k in ['embedding_rmse','energy_score','energy_score_u','coverage90','failure']}
            for t in [50,100,300]}


def main():
    root=Path('adaptive_search_results');w=tail_windows('dev',300,8);out=dict(groups={})
    groups=[(f'checked_p{p}',p) for p in [8,32,128,256]]+[('adapted_probe_read',8),('adapted_full',8)]
    for name,p in groups:
        all_runs=[];per_model={}
        for m in [0,1901,2718,3141]:
            stem=f'{name}_model{m}';doc=json.loads((root/f'{stem}.json').read_text())
            assert len(doc['runs'])==3;fails=0;total=0
            for r in doc['runs']:
                assert r['verification']['violations']==0
                with np.load(root/f'{stem}_roll{r["seed"]}.npz') as a:
                    for k,k0 in [('truth','truth'),('video','video'),('window_start','start')]:
                        np.testing.assert_array_equal(a[k],w[k0])
                    pred=a['prediction'];failed=a['failed'];assert pred.shape==(32,p,300,2)
                    assert np.all(failed[:,:,1:]>=failed[:,:,:-1])
                    fails+=int(failed[:,:,-1].sum());total+=32*p
                    for t in [50,100,300]:
                        x=pred[:,:,t-1];y=w['truth'][:,t-1]
                        emb=np.concatenate([np.sin(x),np.cos(x)],-1)
                        target=np.concatenate([np.sin(y),np.cos(y)],-1)
                        distances=np.linalg.norm(emb-target[:,None],axis=-1).mean(1)
                        np.testing.assert_allclose(np.sqrt(((emb.mean(1)-target)**2).mean()),r['score'][str(t)]['embedding_rmse'],rtol=1e-13,atol=1e-13)
                        r['score'][str(t)]['energy_score_u']=energy_u(r['score'][str(t)]['energy_score'],float(distances.mean()),p)
                        for v in np.unique(w['video']):
                            s=r['per_video'][str(v)][str(t)]
                            s['energy_score_u']=energy_u(s['energy_score'],float(distances[w['video']==v].mean()),p)
            per_model[str(m)]=dict(score=aggregate(doc['runs']),cpu_seconds=float(np.mean([r['stats']['cpu_seconds'] for r in doc['runs']])),
                                   failed_endpoints=fails,total_particles=total,
                                   rng_rmse_std={str(t):float(np.std([r['score'][str(t)]['embedding_rmse'] for r in doc['runs']],ddof=1)) for t in [50,100,300]})
            if m:all_runs.extend(doc['runs'])
        family=dict(score=aggregate(all_runs),cpu_seconds=float(np.mean([r['stats']['cpu_seconds'] for r in all_runs])),
                    failed_endpoints=sum(per_model[str(m)]['failed_endpoints'] for m in [1901,2718,3141]),
                    total_particles=sum(per_model[str(m)]['total_particles'] for m in [1901,2718,3141]),
                    per_video={str(v):aggregate([dict(score=r['per_video'][str(v)]) for r in all_runs]) for v in np.unique(w['video'])})
        out['groups'][name]=dict(particles=p,per_model=per_model,adapted_family=family)
        print(name,'original',per_model['0']['cpu_seconds'],[per_model['0']['score'][str(t)]['embedding_rmse'] for t in [50,100,300]],
              'family',family['cpu_seconds'],[family['score'][str(t)]['embedding_rmse'] for t in [50,100,300]],'failures',family['failed_endpoints'],flush=True)
    out.update(window_alignment_and_rmse_verified=True,
        note='Same four reused DEV videos. Empirical energy V-statistic includes self pairs; energy_score_u removes them. Unbiased population interpretation requires IID samples; shared RNG/rollback coupling is not established IID here. Compare both, not a guaranteed unbiased score. CPU scope includes independent checks, not scoring/compression. Checked has budget_factor10 and unrestricted revision distance; search uses budget300/revision2. This is an offline compute frontier, not equivalent latency/rollback constraints or blind validation. No best optimizer selection.')
    (root/'particle_compute_frontier.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
