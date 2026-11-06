from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .backend import make_backend
from .common import (InvalidState, action_distance, atomic_json, atomic_npz, controls,
                     lock, metrics, selections, write_rows, pareto_indices3)


def case_dir(root, state, offset=0):
    if offset == 0:
        return Path(root)/'cases'/state.key
    return Path(root)/'directions'/state.key/f'offset_{offset:+03d}'


def load_union(folder, config, fingerprint):
    records = {}
    seeds = []
    for seed in config['seeds']:
        d = Path(folder)/f'seed_{seed}'
        if not (d/'done.json').exists():
            raise RuntimeError(f'Missing completed restart: {d}')
        if json.loads((d/'done.json').read_text())['fingerprint'] != fingerprint:
            raise RuntimeError(f'Fingerprint mismatch: {d}')
        z = np.load(d/'evaluations.npz',allow_pickle=False)
        seeds.append((seed,z))
        for i in np.flatnonzero(z['ok']):
            x = z['X'][i]
            key = x.tobytes()
            r = {k:z[k][i] for k in ('power','loads','ws','ti')}
            if key in records:
                # The frozen evaluator must be deterministic across processes.
                if any(not np.allclose(records[key][k],r[k],rtol=1e-9,atol=1e-9) for k in r):
                    raise RuntimeError('Different predictions for identical action across restarts')
                records[key]['seeds'].add(seed)
            else:
                records[key] = {'x':x,**r,'seeds':{seed}}
    return list(records.values()),seeds


def metric_table(records, greedy_loads, pmax):
    return np.asarray([metrics(r['power'],r['loads'],greedy_loads,pmax) for r in records])


def refine(records, engine, state, c, greedy, pmax):
    """Budget/envelope-constrained local refinement; fresh checks decide acceptance.

    A failed SLSQP status never licenses a constraint violation. This step augments
    the archive; final STR/MOR/BAL are reselected using the same payoff rule.
    """
    base_m = metric_table(records,greedy,pmax)
    bounds = controls(c)
    additions,logs = [],[]
    for budget in c['budgets']:
        for tau in c['taus']:
            chosen,anchors = selections(base_m,c,budget,tau)
            if not chosen:
                continue
            ideal,spans = anchors
            for mode,i in chosen.items():
                cache = {}
                def get(x):
                    key = np.asarray(x,float).tobytes()
                    if key not in cache:
                        try:
                            ev = engine.evaluate(state,x)
                            cache[key] = (ev,metrics(ev['power'],ev['loads'],greedy,pmax))
                        except InvalidState:
                            cache[key] = (None,None)
                    return cache[key]
                def obj(x):
                    _,m = get(x)
                    if m is None:
                        return 1e6
                    if mode != 'BAL':
                        return m[1 if mode == 'STR' else 2]
                    q = (m[1:3]-ideal)/spans
                    return q.max()+c['rho_bal']/2*q.sum()
                def cons(x):
                    _,m = get(x)
                    return np.full(3,-1e3) if m is None else np.array([budget-m[0],tau-m[3],tau-m[4]])
                start = time.perf_counter()
                result = minimize(obj,records[i]['x'],method='SLSQP',bounds=bounds.tolist(),
                    constraints=[{'type':'ineq','fun':cons}],
                    options={'maxiter':100,'ftol':1e-9,'eps':1e-3})
                # Keep every valid trial. Screening is repeated on the full union.
                for key,(ev,m) in cache.items():
                    if ev is not None:
                        # The backend also returns geometry as x/y: action wins.
                        additions.append({**ev,'x':np.frombuffer(key,dtype=np.float64).copy(),'seeds':set()})
                ev,m = get(result.x)
                logs.append({'mode':mode,'budget':budget,'tau':tau,'success':bool(result.success),
                    'message':str(result.message),'evaluations':len(cache),
                    'final_feasible':bool(m is not None and (cons(result.x)>=-c['feasibility_tolerance']).all()),
                    'seconds':time.perf_counter()-start})
    return additions,logs


def convergence_rows(seed_data, config, greedy, pmax):
    from pymoo.indicators.hv import HV
    rows = []
    # Fixed scaling/reference across all horizons and restarts in this condition.
    # Only the decision-relevant <=6% loss region is scored. Loads are unitless.
    ref = np.array([1.05,1.5,1.5])
    scale = np.array([max(config['budgets']),1.,1.])
    for seed,z in seed_data:
        valid = z['ok']
        seen = z['seen']
        for horizon in config['checkpoints']:
            mask = valid & ((seen[:,0]>=0) | ((seen[:,1]>=0)&(seen[:,1]<=horizon)) |
                            ((seen[:,2]>=0)&(seen[:,2]<=horizon)))
            idx = np.flatnonzero(mask)
            m = np.asarray([metrics(z['power'][i],z['loads'][i],greedy,pmax) for i in idx])
            region = m[(m[:,0]<=max(config['budgets'])) & (m[:,0]>=-1e-8),:3] / scale
            hv = float(HV(ref_point=ref)(region[pareto_indices3(region)])) if len(region) else 0.
            for budget in config['budgets']:
                for tau in config['taus']:
                    select,_ = selections(m,config,budget,tau)
                    for mode in ('STR','MOR','BAL'):
                        row = {'seed':seed,'generation_budget':horizon,'budget':budget,'tau':tau,
                            'mode':mode,'n_valid_candidates':len(idx),'hypervolume_fixed_reference':hv,
                            'status':'FEASIBLE' if mode in select else 'NO_FEASIBLE_CANDIDATE_FOUND'}
                        if mode in select:
                            values = m[select[mode]]
                            row.update(zip(('power_loss','JS','JM','ES','EM'),values.tolist()))
                        rows.append(row)
    return rows


def merge(state, config, backend, root, fingerprint, offset=0, polish=False):
    folder = case_dir(root,state,offset)
    with lock(folder/'merge.lock'):
        records,seed_data = load_union(folder,config,fingerprint)
        engine = make_backend(backend,config)
        greedy_ev = engine.evaluate(state,np.zeros(10))
        greedy = greedy_ev['loads']
        pmax = max(float(r['power'].sum()) for r in records)
        unpolished_pmax = pmax
        logs = []
        if polish:
            added,logs = refine(records,engine,state,config,greedy,pmax)
            dedup = {r['x'].tobytes():r for r in records}
            for r in added:
                dedup.setdefault(r['x'].tobytes(),r)
            records = list(dedup.values())
            pmax = max(float(r['power'].sum()) for r in records)
        m = metric_table(records,greedy,pmax)
        imax = int(np.argmax([r['power'].sum() for r in records]))
        greedy_index = next(i for i,r in enumerate(records) if np.array_equal(r['x'],np.zeros(10)))
        choices = [('Greedy',None,None,greedy_index,None),('Pbest',None,None,imax,None)]
        missing = []
        for budget in config['budgets']:
            for tau in config['taus']:
                select,anchors = selections(m,config,budget,tau)
                if not select:
                    missing.append({'budget':budget,'tau':tau,'status':'NO_FEASIBLE_CANDIDATE_FOUND'})
                for mode,i in select.items():
                    choices.append((mode,budget,tau,i,anchors))
        selected,turbines,verified = [],[],[]
        for mode,budget,tau,i,anchors in choices:
            r = records[i]
            ev = engine.evaluate(state,r['x'])  # intentionally bypass optimizer memoization
            v = metrics(ev['power'],ev['loads'],greedy,pmax)
            if not np.allclose(v,m[i],rtol=1e-8,atol=1e-9):
                raise RuntimeError('Selected action does not reproduce its archive metrics')
            if budget is not None and (v[0]>budget+config['feasibility_tolerance'] or
                    max(v[3:5])>tau+config['feasibility_tolerance']):
                raise RuntimeError('Selected action fails final independent constraint check')
            tag = mode if budget is None else f'{mode}{round(budget*100)}_tau{tau:.3f}'
            row = {'farm':state.farm,'centroid':state.centroid,'wd':state.wd,'strategy':tag,
                'mode':mode,'budget':budget,'tau':tau,'power_kw':float(ev['power'].sum()),'pbest_kw':pmax,
                'reference':'best_found_feasible_power','status':'VERIFIED',
                **dict(zip(('power_loss','JS','JM','ES','EM'),v.tolist()))}
            if anchors:
                ideal,spans = anchors
                q = (v[1:3]-ideal)/spans
                row['JB'] = float(q.max()+config['rho_bal']/2*q.sum())
            row.update({f'yaw_T{j+1}':float(r['x'][j]) for j in range(5)})
            row.update({f'pitch_T{j+1}':float(r['x'][j+5]) for j in range(5)})
            selected.append(row)
            verified.append({'strategy':tag,'action':r['x'].tolist(),
                'budget':budget,'tau':tau,'metrics':v.tolist()})
            ratio = ev['loads']/greedy
            for j in range(5):
                turbines.append({'farm':state.farm,'centroid':state.centroid,'strategy':tag,'turbine':j+1,
                    'yaw_deg':float(r['x'][j]),'pitch_deg':float(r['x'][j+5]),
                    'power_kw':float(ev['power'][j]),'ws_eff':float(ev['ws'][j]),'ti_eff_pct':float(ev['ti'][j]),
                    **{name:float(ev['loads'][j,k]) for k,name in enumerate(('DEL_TB','DEL_BR','DEL_YB','DEL_Fmax'))},
                    **{name:float(ratio[j,k]) for k,name in enumerate(('ratio_TB','ratio_BR','ratio_YB','ratio_Fmax'))}})
        # Diagnostic front only: the complete archive remains available for selection.
        nd = pareto_indices3(m[:,:3])
        out = folder/'merged'
        out.mkdir(exist_ok=True)
        atomic_npz(out/'archive.npz',X=np.asarray([r['x'] for r in records]),metrics=m,
            power=np.asarray([r['power'] for r in records]),loads=np.asarray([r['loads'] for r in records]),
            nondominated_indices=nd)
        write_rows(out/'selected.csv',selected)
        write_rows(out/'selected_turbines.csv',turbines)
        write_rows(out/'convergence.csv',convergence_rows(seed_data,config,greedy,pmax))
        write_rows(out/'missing_selections.csv',missing)
        atomic_json(out/'selection.json',{'fingerprint':fingerprint,'state':state.__dict__,
            'pbest_kw':pmax,'unpolished_pbest_kw':unpolished_pmax,'greedy_loads':greedy.tolist(),
            'selected':verified,'missing':missing,'polish':polish,'polish_logs':logs,
            'backend':backend,'synthetic':backend=='analytic',
            'note':'No feasible candidate found is not proof of infeasibility; Pbest is not a global optimality certificate.'})
        print(f'Merged {state.key}: {len(records)} candidates; {len(selected)} selections; {len(missing)} empty feasible sets',flush=True)


def scan(state, config, backend, root, fingerprint, mode='fixed-offset', yaw_error_sign=None):
    if mode == 'fixed-heading' and yaw_error_sign not in (-1,1):
        raise ValueError('Fixed-heading scan requires a VERIFIED yaw-error sign (+1 or -1)')
    center_path = case_dir(root,state)/'merged/selection.json'
    center = json.loads(center_path.read_text())
    if center['fingerprint'] != fingerprint:
        raise RuntimeError('Centroid selection fingerprint mismatch')
    engine = make_backend(backend,config)
    rows = []
    for delta in range(-5,6):
        perturbed = state.shifted(delta)
        try:
            greedy = engine.evaluate(perturbed,np.zeros(10))
        except InvalidState as exc:
            rows.append({'farm':state.farm,'centroid':state.centroid,'strategy':'ALL',
                         'delta_wd':delta,'mode':mode,'status':'INVALID_GREEDY_REFERENCE','reason':str(exc)})
            continue
        reference_path = case_dir(root,state,delta)/'merged/selection.json'
        reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
        if reference and reference['fingerprint'] != fingerprint:
            raise RuntimeError('Direction selection fingerprint mismatch')
        pbest = reference['pbest_kw'] if reference else None
        # Keep Greedy, Pbest, and the six tau=1.025 publication candidates.
        for entry in center['selected']:
            if entry['tau'] is not None and abs(entry['tau']-1.025)>1e-9:
                continue
            action = np.asarray(entry['action'],float)
            if mode == 'fixed-heading':
                action[:5] += yaw_error_sign*delta
            row = {'farm':state.farm,'centroid':state.centroid,'strategy':entry['strategy'],
                'delta_wd':delta,'actual_wd':perturbed.wd,'mode':mode,
                'yaw_error_sign':yaw_error_sign,'reference':'greedy at perturbed actual WD'}
            b = controls(config)
            if (action<b[:,0]).any() or (action>b[:,1]).any():
                row.update(status='OUTSIDE_TRAINED_ACTION_DOMAIN',reason='Actual yaw offset leaves surrogate support; not clipped')
                rows.append(row)
                continue
            try:
                ev = engine.evaluate(perturbed,action)
                m = metrics(ev['power'],ev['loads'],greedy['loads'],pbest or float(ev['power'].sum()))
                row.update(status='VALID',power_kw=float(ev['power'].sum()),JS=m[1],JM=m[2],ES=m[3],EM=m[4],
                    greedy_power_kw=float(greedy['power'].sum()),
                    power_loss=m[0] if pbest else '',pbest_kw=pbest if pbest else '',
                    envelope_excess=max(0.,float(max(m[3:5]))-1.025),
                    budget_excess=max(0.,float(m[0])-entry['budget']) if pbest and entry['budget'] is not None else '')
                for j in range(5):
                    row[f'ws_T{j+1}'],row[f'ti_T{j+1}'] = ev['ws'][j],ev['ti'][j]
                if reference:
                    match = next((e for e in reference['selected'] if e['strategy']==entry['strategy']),None)
                    if match:
                        row['normalized_action_distance_to_reoptimized'] = action_distance(action,np.array(match['action']),config)
                        row['JS_minus_reoptimized'] = float(m[1]-match['metrics'][1])
                        row['JM_minus_reoptimized'] = float(m[2]-match['metrics'][2])
            except InvalidState as exc:
                row.update(status='INVALID_MODEL_DOMAIN',reason=str(exc))
            rows.append(row)
    write_rows(case_dir(root,state)/f'scan_{mode}.csv',rows)
    print(f'Saved {state.key} {mode} scan: {len(rows)} rows',flush=True)
