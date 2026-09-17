"""Aggregate prespecified fixed-candidate RNG audits; descriptive only."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('adaptive_search_results');sets=[]
    for seed in [1901,2718,3141]:
        report=json.loads((root/f'writer_ranking_audit_seed{seed}.json').read_text())
        assert report['exact_original_cost_replay'] and report['source_unchanged']
        sets.extend(dict(optimization_seed=seed,iteration=s['iteration'],**s['summary']) for s in report['sets'])
    assert len(sets)==9
    moves=[s for s in sets if s['original_chosen']!=0]
    result=dict(sets=sets,total_sets=len(sets),original_moves=len(moves),
        original_move_mean_difference=float(np.mean([s['original_difference'] for s in moves])),
        fresh_move_mean_difference=float(np.mean([s['chosen_mean_difference'] for s in moves])),
        harmful_original_moves_on_fresh_mean=sum(s['chosen_mean_difference']>0 for s in moves),
        mean_original_winner_replica_agreement=float(np.mean([s['same_winner_fraction'] for s in sets])),
        original_winner_matches_fresh_mean=sum(s['original_chosen']==s['mean_winner'] for s in sets),
        group3_agreement=sum(s['group3_winners'][0]==s['group3_winners'][1] for s in sets),
        single_seed_split_agreement=sum(s['fresh_winners'][0]==s['fresh_winners'][3] for s in sets),
        group3_cross_mean_difference=float(np.mean([d for s in sets for d in s['group3_cross_evaluation']])),
        note='Descriptive9sets,7or fewer acceptedmoves, overlapping videos/windows and RNG streams. No formal independent-set inference.3seed averaging comparison uses same6runs, not optimized retraining/equal-budget learning experiment.')
    (root/'writer_ranking_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='sets'},indent=2))


if __name__=='__main__':main()
