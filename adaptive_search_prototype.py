"""Small AlphaGo-style lookahead prototype for v20_rnn_mixture.

This is deliberately isolated from the frozen package.  Every virtual edge
keeps the required order: read event probabilities -> choose an event ->
choose q' inside that event -> execute F(q,q').  Search only uses generated
states; it never receives truth.  The root edge is the only edge committed to
the returned trajectory.

This first version uses variable-depth beam lookahead with a heuristic value
(policy log-probability, mixture margin, and diagnosed-q contradiction).  It
is an experiment toward PUCT/MCTS, not yet a trained AlphaZero value network.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import OrderedDict

import numpy as np

from v20_rnn_mixture.engine.checker import MixtureChecker
from v20_rnn_mixture.engine.data import tail_windows
from v20_rnn_mixture.engine.dynamics import Dynamics
from v20_rnn_mixture.engine.evaluate import metrics, tracking
from v20_rnn_mixture.engine.event import EventMachine


@dataclass
class Node:
    history: np.ndarray
    q: int
    hidden: np.ndarray
    depth: int
    score: float
    root_event: int
    root_q: int
    event: int
    next_q: int
    path_events: tuple[int, ...] = field(default_factory=tuple)
    path_q: tuple[int, ...] = field(default_factory=tuple)
    check_sum: float = 0.0
    contradictions: int = 0
    check_middle: bool = False


class AdaptiveBeam:
    """Variable-depth event-first beam search over one particle state."""

    def __init__(self, checkpoint="gru_1901.npz", checker="mixture", alpha=0.01,
                 event_top=2, next_top=2, beam_per_root=3, min_depth=2,
                 max_depth=4, stability_rounds=1, temperature=0.0,
                 check_weight=0.08, contradiction_weight=0.35,
                 widen_on_empty=False, soft_check=False, value_model=None, value_weight=1.,
                 min_stop_support=1, search_interval=1, uncertainty_gap=0., rule_cache_size=0,
                 depth_invariant_temperature=False, shared_writer=False, boundary_repair=False, policy_adapter=None, commit_probe=False):
        self.base = Dynamics()
        self.shared_writer = None
        self.boundary_repair = boundary_repair
        self.commit_probe = commit_probe
        if shared_writer:
            from shared_candidate_writer import SharedCandidateWriter
            self.shared_writer = SharedCandidateWriter(self.base)
        self.machine = EventMachine(self.base, Path(__file__).resolve().parent /
                                    "v20_rnn_mixture" / "models" / checkpoint)
        if policy_adapter:
            from stateless_policy_adapter import StatelessPolicyAdapter
            self.machine=StatelessPolicyAdapter(self.machine,policy_adapter)
        self.checker = MixtureChecker(checker, alpha)
        self.event_top = int(event_top)
        self.next_top = int(next_top)
        self.beam_per_root = int(beam_per_root)
        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        self.stability_rounds = int(stability_rounds)
        self.temperature = float(temperature)
        self.check_weight = float(check_weight)
        self.contradiction_weight = float(contradiction_weight)
        self.widen_on_empty = widen_on_empty
        self.soft_check = soft_check
        self.value_weight = value_weight
        self.min_stop_support = int(min_stop_support)
        self.search_interval = int(search_interval)
        self.uncertainty_gap = float(uncertainty_gap)
        self.rule_cache_size = int(rule_cache_size)
        self.depth_invariant_temperature = depth_invariant_temperature
        self._rule_cache = OrderedDict()
        if self.search_interval < 1:
            raise ValueError('search_interval must be positive')
        self.value_model = None
        if value_model:
            from train_search_value import ValueModel
            self.value_model = ValueModel(value_model)
        self.audit = dict(expansions=0, empty_narrow=0, rescued=0,
                          empty_full=0, rejected_edges=0, rule_cache_hits=0, rule_cache_misses=0,
                          viability_probes=0,viability_full_probes=0,viability_rejected_roots=0)
        if self.min_depth < 1 or self.max_depth < self.min_depth:
            raise ValueError("Require 1 <= min_depth <= max_depth")

    @staticmethod
    def _top(a, k):
        k = min(int(k), len(a))
        return np.argsort(a)[-k:][::-1]

    def _read(self, node):
        pe, trans, read_result = self.machine.read(
            node.history, np.asarray([node.q]), {"hidden": node.hidden})
        return pe[0], trans[0], read_result["read_hidden"]

    def _child(self, parent, e, r, pe, trans, read_hidden):
        y = self.base.execute_rule(parent.history, np.asarray([parent.q]),
                                   np.asarray([r]))
        y = y[0]
        if not np.isfinite(y).all() or np.max(np.abs(y-parent.history[0, -1])) > 20:
            return None

        score_penalty = 0.0
        check_sum = parent.check_sum
        # The first generated point has no generated middle point yet.  Once
        # depth>=1, the new right point checks the previous generated middle.
        if parent.depth >= 1 or parent.check_middle:
            left = parent.history[:, :-1]
            middle = parent.history[:, -1]
            right = y[None]
            reject, score = self.checker.reject(left, middle, right,
                                                np.asarray([parent.q]))
            if bool(reject[0]) and not self.soft_check:
                return None
            score_penalty = self.check_weight * float(score[0])
            check_sum += float(score[0])

        nh = np.concatenate([parent.history[:, 1:], y[None, None]], axis=1)
        diagnosed = int(self.base.state_from_history(nh)[0][0])
        contradiction = int(diagnosed != int(r))
        logp = math.log(max(float(pe[e]), 1e-12)) + math.log(max(float(trans[e, r]), 1e-12))
        reward = logp - score_penalty - self.contradiction_weight * contradiction
        return Node(
            history=nh, q=int(r), hidden=np.asarray(read_hidden).copy(),
            depth=parent.depth + 1, score=parent.score + reward,
            root_event=int(e if parent.depth == 0 else parent.root_event),
            root_q=int(r if parent.depth == 0 else parent.root_q),
            event=int(e), next_q=int(r),
            path_events=parent.path_events + (int(e),),
            path_q=parent.path_q + (int(r),),
            check_sum=check_sum, contradictions=parent.contradictions + contradiction)

    def _execute_candidates(self, history, q, rs):
        # F is independent of GRU hidden/event. Reuse only exact numeric
        # inputs; keep event identities/probabilities in the search edges.
        key = (history.shape, history.dtype.str, history.tobytes(), int(q), tuple(rs)) if self.rule_cache_size>0 else None
        if key is not None and key in self._rule_cache:
            self.audit['rule_cache_hits'] += 1
            value=self._rule_cache.pop(key); self._rule_cache[key]=value
            return value
        self.audit['rule_cache_misses'] += 1
        value=(self.shared_writer.execute(history,q,rs) if self.shared_writer is not None else
               self.base.execute_rule(np.repeat(history,len(rs),axis=0),np.full(len(rs),q,dtype=int),rs))
        if key is not None:
            value.setflags(write=False)
            self._rule_cache[key]=value
            if len(self._rule_cache)>self.rule_cache_size:self._rule_cache.popitem(last=False)
        return value

    def _expand(self, node, full=False):
        self.audit['expansions'] += 1
        pe, trans, read_hidden = self._read(node)
        pairs = [(int(e), int(r)) for e in self._top(pe, len(pe) if full else self.event_top)
                 for r in self._top(trans[e], len(trans[e]) if full else self.next_top)]
        if not pairs:
            return []

        # Execute all event/q' candidates for this node in one batch.  The
        # frozen F is unchanged; this only removes hundreds of tiny
        # ``features.design``/einsum calls from the Python loop.
        n = len(pairs)
        histories = np.repeat(node.history, n, axis=0)
        qs = np.full(n, int(node.q), dtype=int)
        rs = np.asarray([r for _, r in pairs], dtype=int)
        ys = self._execute_candidates(node.history,node.q,rs)
        valid = np.isfinite(ys).all(axis=1)
        valid &= np.max(np.abs(ys - node.history[0, -1]), axis=1) <= 20

        scores = np.zeros(n, dtype=float)
        rejected = np.zeros(n, dtype=bool)
        if node.depth >= 1 or node.check_middle:
            left = np.repeat(node.history[:, :-1], n, axis=0)
            middle = np.repeat(node.history[:, -1], n, axis=0)
            rejected, scores = self.checker.reject(left, middle, ys, qs)
            rejected = np.asarray(rejected, dtype=bool)
            scores = np.asarray(scores, dtype=float)

        self.audit['rejected_edges'] += int(rejected.sum())
        keep = valid & np.isfinite(scores) if self.soft_check else valid & ~rejected
        if not keep.any():
            self.audit['empty_full' if full else 'empty_narrow'] += 1
            if self.widen_on_empty and not full:
                expanded = self._expand(node, full=True)
                self.audit['rescued'] += int(bool(expanded))
                return expanded
            return []
        nh_all = np.concatenate([histories[:, 1:, :], ys[:, None, :]], axis=1)
        diagnosed = self.base.state_from_history(nh_all)[0]
        out = []
        for i, (e, r) in enumerate(pairs):
            if not keep[i]:
                continue
            contradiction = int(int(diagnosed[i]) != r)
            logp = math.log(max(float(pe[e]), 1e-12)) + math.log(max(float(trans[e, r]), 1e-12))
            reward = logp - self.check_weight * float(scores[i]) - self.contradiction_weight * contradiction
            out.append(Node(
                history=nh_all[i:i + 1].copy(), q=r, hidden=np.asarray(read_hidden).copy(),
                depth=node.depth + 1, score=node.score + reward,
                root_event=e if node.depth == 0 else node.root_event,
                root_q=r if node.depth == 0 else node.root_q,
                event=e, next_q=r,
                path_events=node.path_events + (e,), path_q=node.path_q + (r,),
                check_sum=node.check_sum + float(scores[i]),
                contradictions=node.contradictions + contradiction))
        return out

    def _root_options(self, history, q, hidden, check_root=True, full=False):
        root = Node(history=history.copy(), q=int(q), hidden=hidden.copy(), depth=0,
                    score=0.0, root_event=-1, root_q=-1, event=-1, next_q=-1,
                    check_middle=check_root)
        return self._expand(root,full=full)

    def _value(self, node):
        # Same-depth comparisons are primary; normalization makes diagnostics
        # interpretable when the adaptive search stops at different depths.
        score = node.score / max(node.depth, 1)
        if self.value_model is not None:
            score -= self.value_weight * self.value_model.predict(self.base, node)
        return score

    def search(self, history, q, hidden, rng, check_root=True, banned_q=(), allow_lookahead=True, full_root=False):
        roots = self._root_options(history, q, hidden, check_root, full_root)
        roots = [node for node in roots if node.q not in banned_q]
        if not roots:
            return None, {"depth": 0, "roots": 0, "stable": False, "root_gap": None}
        # Shallow calls still validate the actual middle. Uncertainty can
        # promote a scheduled shallow call; this is an ablation heuristic.
        root_scores = sorted((node.score for node in roots), reverse=True)
        uncertain = self.uncertainty_gap > 0 and (len(root_scores)<2 or root_scores[0]-root_scores[1]<self.uncertainty_gap)
        depth_limit = self.max_depth if allow_lookahead or uncertain else 1
        groups = {i: [root] for i, root in enumerate(roots)}
        previous = None
        stable = 0
        root_values = np.full(len(roots), -np.inf)
        depth_done = 1
        for depth in range(1, depth_limit + 1):
            if depth > 1:
                for i, nodes in list(groups.items()):
                    expanded = []
                    for node in nodes:
                        expanded.extend(self._expand(node))
                    if expanded:
                        expanded.sort(key=self._value, reverse=True)
                        groups[i] = expanded[:self.beam_per_root]
                    else:
                        groups[i] = []
            for i, nodes in groups.items():
                root_values[i] = max((self._value(x) for x in nodes), default=-np.inf)
            order = np.argsort(root_values)[::-1]
            if len(order) and np.isfinite(root_values[order[0]]):
                winner = int(order[0])
                if previous == winner:
                    stable += 1
                else:
                    stable = 0
                previous = winner
            depth_done = depth
            if len(order) > 1 and np.isfinite(root_values[order[0]]) and np.isfinite(root_values[order[1]]):
                gap = root_values[order[0]] - root_values[order[1]]
            else:
                gap = -np.inf
            # Count distinct q paths, not duplicate event edges. This is a
            # finite-horizon support heuristic, not a feasibility guarantee.
            support = len({x.path_q for x in groups.get(previous, [])})
            if depth >= self.min_depth and stable >= self.stability_rounds and gap > 0.02 and support >= self.min_stop_support:
                break

        finite = np.isfinite(root_values)
        if not finite.any():
            return None, {"depth": depth_done, "roots": len(roots), "stable": False, "root_gap": None}
        vals = root_values.copy()
        witness_q=None
        while True:
            if self.temperature > 0:
                # Keep the original SCORING depth/temperature even if a
                # later viability-only probe examines a second edge.
                temp = self.temperature * self.max_depth / depth_done if self.depth_invariant_temperature else self.temperature
                z = (vals - np.nanmax(vals)) / temp
                probs = np.exp(np.clip(z, -60, 0)); probs[~finite] = 0
                probs /= probs.sum()
                chosen = int(rng.choice(len(roots), p=probs))
            else:
                chosen = int(np.nanargmax(vals))
            if not self.commit_probe or depth_done>1:break
            self.audit['viability_probes']+=1
            children=self._expand(roots[chosen])
            if not children:
                self.audit['viability_full_probes']+=1
                children=self._expand(roots[chosen],full=True)
            if children:
                witness_q=int(children[0].q)
                break
            # No e-specific commit state: equal root q has the same future.
            # Remove those roots locally, without changing parent GRU memory.
            removed=np.array([r.q==roots[chosen].q for r in roots])&finite
            self.audit['viability_rejected_roots']+=int(removed.sum())
            vals[removed]=-np.inf;finite=np.isfinite(vals)
            if not finite.any():
                return None,dict(depth=2,scoring_depth=depth_done,roots=len(roots),stable=False,root_gap=None,
                                 viability_checked=True,viability_exhausted=True)
        node = roots[chosen]
        # Use the best continuation only to choose root; commit the root edge.
        best = max(groups.get(chosen, [node]), key=self._value)
        info = dict(depth=max(depth_done,2 if witness_q is not None else 1),scoring_depth=depth_done,
                    viability_checked=(depth_done>1 or witness_q is not None),viability_witness_q=witness_q,
                    roots=len(roots), stable=(stable >= self.stability_rounds),
                    root_event=node.root_event, root_q=node.root_q,
                    root_value=float(root_values[chosen]), root_gap=float(np.sort(vals[finite])[-1] - np.sort(vals[finite])[-2]) if finite.sum() > 1 else None,
                    continuation_value=float(self._value(best)),
                    continuation_events=list(best.path_events), continuation_q=list(best.path_q),
                    check_sum=float(best.check_sum), contradictions=int(best.contradictions))
        return node, info


def rollout(searcher, history, horizon, particles=8, seed=1729, rollback_budget=0, rollback_window=None):
    history = np.asarray(history, dtype=float)
    q0, mem0 = searcher.machine.initialize(history)
    rng = np.random.default_rng(seed)
    pred = np.zeros((1, particles, horizon, 2), dtype=float)
    failed = np.zeros((1, particles, horizon), dtype=bool)
    diagnostics = []
    for particle in range(particles):
        h = history.copy(); q = int(q0[0]); hidden = mem0["hidden"].copy()
        stack = [(h, q, hidden, set())]
        t = 0; rewinds = 0; frontier = 0
        revisions = np.zeros(horizon,dtype=int)
        while t < horizon:
            h, q, hidden, banned = stack[t]
            node, info = searcher.search(h, q, hidden, rng, check_root=(t > 0), banned_q=banned,
                                         allow_lookahead=(t % getattr(searcher,'search_interval',1)==0))
            within_window = rollback_window is None or frontier-(t-1) <= rollback_window
            repair_attempted = False
            if node is None and t>0 and not within_window and getattr(searcher,'boundary_repair',False):
                repair_attempted = True
                node,info = searcher.search(h,q,hidden,rng,check_root=True,banned_q=banned,
                                            allow_lookahead=True,full_root=True)
            info.update(particle=particle, step=t, rollback=False, rollback_distance=0, rewritten=False,
                        repair_attempted=repair_attempted,repair_success=repair_attempted and node is not None)
            diagnostics.append(info)
            if node is None:
                if t > 0 and rewinds < rollback_budget and within_window:
                    # Restore the actual parent state and ban the dead child
                    # only at that parent; later descendants are discarded.
                    dead_q = q
                    stack.pop(); t -= 1; rewinds += 1
                    stack[t][3].add(dead_q)
                    info['rollback'] = True
                    info['rollback_distance'] = frontier-t
                    continue
                failed[0, particle, t:] = True
                pred[0, particle, t:] = h[0, -1]
                break
            y = node.history[0, -1]
            info['rewritten'] = bool(t < frontier)
            revisions[t] += 1
            info['step_revision_count'] = int(revisions[t]-1)
            pred[0, particle, t] = y
            h, q, hidden = node.history, node.q, node.hidden
            stack.append((h, q, hidden, set()))
            t += 1
            frontier = max(frontier,t)
    return pred, failed, diagnostics


def make_searcher(args):
    return AdaptiveBeam(max_depth=args.max_depth, min_depth=args.min_depth,
                            event_top=args.event_top, next_top=args.next_top,
                            beam_per_root=args.beam_width,
                            stability_rounds=args.stability_rounds,
                            temperature=args.temperature,
                            check_weight=args.check_weight,
                            contradiction_weight=args.contradiction_weight,
                            widen_on_empty=args.widen_on_empty, soft_check=args.soft_check,
                            value_model=args.value_model, value_weight=args.value_weight,
                            min_stop_support=args.min_stop_support,
                            search_interval=args.search_interval, uncertainty_gap=args.uncertainty_gap,
                            rule_cache_size=args.rule_cache_size,
                            depth_invariant_temperature=args.depth_invariant_temperature,
                            shared_writer=args.shared_writer,boundary_repair=args.boundary_repair,
                            policy_adapter=getattr(args,'policy_adapter',None),commit_probe=getattr(args,'commit_probe',False))


def run_window(task):
    cpu_start=time.process_time()
    i, h, args = task
    searcher = make_searcher(args)
    actual_particles = 1 if args.temperature == 0 else args.particles
    p,f,d = rollout(searcher,h[None],args.steps,actual_particles,args.seed+i,args.rollback_budget,args.rollback_window)
    if actual_particles != args.particles:
        p=np.repeat(p,args.particles,axis=1); f=np.repeat(f,args.particles,axis=1)
    audit=dict(searcher.audit,worker_cpu_seconds=time.process_time()-cpu_start)
    return p[0],f[0],d,audit


def run(args):
    if args.workers < 1 or args.particles < 1:
        raise ValueError('workers and particles must be positive')
    if args.rollback_budget < 0 or (args.rollback_window is not None and args.rollback_window < 0):
        raise ValueError('rollback budget/window must be nonnegative')
    window_horizon = args.window_horizon or args.steps
    if window_horizon < args.steps:
        raise ValueError('window-horizon must cover steps')
    windows = tail_windows(args.split, window_horizon, args.per_video)
    windows['truth'] = windows['truth'][:, :args.steps]
    all_pred=[]; all_failed=[]; all_diag=[]; audit={}; start=time.perf_counter()
    tasks=[(i,h,args) for i,h in enumerate(windows['history'])]
    pool=ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) if args.workers>1 else None
    try:
        results=pool.map(run_window,tasks) if pool else map(run_window,tasks)
        for i,(p,f,d,a) in enumerate(results):
            all_pred.append(p); all_failed.append(f); all_diag.extend(d)
            for k,v in a.items():audit[k]=audit.get(k,0)+v
            if (i+1)%4==0: print(f"completed {i+1}/{len(tasks)}",flush=True)
    finally:
        if pool:pool.shutdown()
    pred=np.asarray(all_pred); failed=np.asarray(all_failed)
    score, arrays=metrics(pred,windows['truth'],failed)
    result=dict(config={k:v for k,v in vars(args).items()}, score=score,
                terminal_right_generated=False,
                completion_note='Legacy H-output runner: no committed extra right point certifies the endpoint. Use evaluate_adapted_checked.py for full-horizon central-check verification.',
                tracking=tracking(arrays['angular_errors']), seconds=time.perf_counter()-start,
                n_windows=len(windows['history']), particles=args.particles,
                failed_fraction=float(failed.mean()), audit=audit,
                rollbacks=sum(d.get('rollback',False) for d in all_diag),
                boundary_repairs=dict(attempts=sum(d.get('repair_attempted',False) for d in all_diag),
                                      successes=sum(d.get('repair_success',False) for d in all_diag)),
                revision=dict(max_rollback_distance=max((d.get('rollback_distance',0) for d in all_diag),default=0),
                              rewritten_steps=sum(d.get('rewritten',False) for d in all_diag),
                              max_step_revisions=max((d.get('step_revision_count',0) for d in all_diag),default=0)),
                search=dict(mean_depth=float(np.mean([d['depth'] for d in all_diag])),
                            max_depth=int(max(d['depth'] for d in all_diag)),
                            stable_fraction=float(np.mean([d['stable'] for d in all_diag])),
                            mean_roots=float(np.mean([d['roots'] for d in all_diag])),
                    mean_root_gap=float(np.mean([d['root_gap'] for d in all_diag if d.get('root_gap') is not None])) if any(d.get('root_gap') is not None for d in all_diag) else None)
                )
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    np.savez_compressed(out.with_suffix('.npz'),prediction=pred,failed=failed,truth=windows['truth'],history=windows['history'],video=windows['video'],window_start=windows['start'],**arrays)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=['dev','test'],default='dev')
    ap.add_argument('--steps',type=int,default=50)
    ap.add_argument('--per-video',type=int,default=2)
    ap.add_argument('--particles',type=int,default=2)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--search-interval',type=int,default=1)
    ap.add_argument('--uncertainty-gap',type=float,default=0.)
    ap.add_argument('--rule-cache-size',type=int,default=0)
    ap.add_argument('--depth-invariant-temperature',action='store_true')
    ap.add_argument('--shared-writer',action='store_true')
    ap.add_argument('--boundary-repair',action='store_true')
    ap.add_argument('--policy-adapter')
    ap.add_argument('--commit-probe',action='store_true')
    ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--min-depth',type=int,default=2)
    ap.add_argument('--max-depth',type=int,default=4)
    ap.add_argument('--event-top',type=int,default=2)
    ap.add_argument('--next-top',type=int,default=2)
    ap.add_argument('--beam-width',type=int,default=3)
    ap.add_argument('--stability-rounds',type=int,default=1)
    ap.add_argument('--temperature',type=float,default=0.0)
    ap.add_argument('--check-weight',type=float,default=0.08)
    ap.add_argument('--contradiction-weight',type=float,default=0.35)
    ap.add_argument('--widen-on-empty', action='store_true')
    ap.add_argument('--soft-check', action='store_true')
    ap.add_argument('--value-model')
    ap.add_argument('--value-weight', type=float, default=1.)
    ap.add_argument('--min-stop-support', type=int, default=1)
    ap.add_argument('--window-horizon', type=int)
    ap.add_argument('--rollback-budget', type=int, default=0,
                    help='Bounded OFFLINE path revision; zero preserves irrevocable rollout')
    ap.add_argument('--rollback-window',type=int,
                    help='Optional maximum withdrawal behind generated frontier; limits revision latency')
    ap.add_argument('--output',default='adaptive_search_results/prototype.json')
    run(ap.parse_args())


if __name__=='__main__': main()
