"""Descriptive fixed-budget grouped-versus-pooled estimator comparison."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results');sets=[];old_variances=[]
    for seed in [1901,2718,3141]:
        audit=json.loads((root/f'particle_grouping_seed{seed}.json').read_text())
        old=json.loads((root/f'writer_ranking_audit_seed{seed}.json').read_text())
        for item,previous in zip(audit['sets'],old['sets']):
            assert item['iteration']==previous['iteration']
            oldcost=np.array(previous['fresh_costs']);old_variances.extend((oldcost[:,1:]-oldcost[:,:1]).var(0,ddof=1))
            methods=item['summary'];pooled=np.array(methods['pooled12']['costs']);grouped=np.array(methods['mean3x4']['costs'])
            out=dict(seed=seed,iteration=item['iteration'],
                     pooled_variance=methods['pooled12']['paired_variance'],
                     grouped_variance=methods['mean3x4']['paired_variance'],
                     identical_replica_winners=int((np.argmin(pooled,1)==np.argmin(grouped,1)).sum()))
            for label,cost in [('pooled',pooled),('grouped',grouped)]:
                winners=np.argmin(cost,1)
                out[label+'_half_winner_agreement']=bool(np.argmin(cost[:3].mean(0))==np.argmin(cost[3:].mean(0)))
                # Evaluate chosen candidate on OTHER replicas using common pooled reference.
                out[label+'_leave_replica_out_delta']=float(np.mean([
                    (pooled[np.arange(6)!=i,winners[i]]-pooled[np.arange(6)!=i,0]).mean() for i in range(6)]))
            sets.append(out)
    pv=np.array([v for s in sets for v in s['pooled_variance']]);gv=np.array([v for s in sets for v in s['grouped_variance']])
    result=dict(sets=sets,paired_comparisons=len(pv),pooled_lower_variance_comparisons=int((pv<gv).sum()),
                mean_pooled_variance=float(pv.mean()),mean_grouped_variance=float(gv.mean()),
                ratio_of_mean_variances=float(pv.mean()/gv.mean()),
                pooled_vs_old_singleP4_variance_ratio=float(pv.mean()/np.mean(old_variances)),
                identical_replica_winners=sum(s['identical_replica_winners'] for s in sets),total_replica_winners=len(sets)*6,
                half_winner_agreement={label:sum(s[label+'_half_winner_agreement'] for s in sets) for label in ['pooled','grouped']},
                leave_replica_out_delta={label:float(np.mean([s[label+'_leave_replica_out_delta'] for s in sets])) for label in ['pooled','grouped']},
                note='Equal trajectory budget only for pooled12 vsmean3x4. OldsingleP4 is3x cheaper/different RNG, descriptive scaling reference.6replicates per set; correlated candidate sets, no formal significance. Leave-replica-out decisions evaluated on pooled costs of other5replicas, not independent-video validation.')
    (root/'particle_grouping_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sets'},indent=2))


if __name__=='__main__':main()
