"""Descriptive regrouping of existing raw gradients; no new evidence draws."""
import json
from pathlib import Path
import numpy as np
from audit_source_q_gradient import stability


def main():
    root=Path('adaptive_search_results');r=json.loads((root/'score_control_variate_audit.json').read_text())
    g=np.array([x['gradients'] for phase in ['fit','evaluation'] for x in r['phases'][phase]]).mean(1)
    assert g.shape==(16,8,2);summary={}
    for n in [1,2,4,8]:
        grouped=g.reshape(16//n,n,8,2).mean(1)
        np.testing.assert_allclose(grouped.mean(0),g.mean(0),rtol=1e-12,atol=1e-14)
        summary[str(n)]=dict(table=stability(grouped),shared=stability(grouped.sum(1)))
    report=dict(summary=summary,note='Post-hoc adjacent grouping of same16raw seed gradients; exact same overall mean/no new samples/no training. n8 leaves only2group observations,too weak for covariance conclusions. Does not disprove variance reduction from genuinely additional independent samples; do not select an optimizer batch size from this report.')
    (root/'control_gradient_grouping.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
