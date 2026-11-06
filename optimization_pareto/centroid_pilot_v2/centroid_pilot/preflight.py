from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .backend import make_backend, target_order
from .common import atomic_json, write_rows


def inspect_models(config):
    """Metadata only, no model arrays or training-data contents printed."""
    for key in ('base_model','sea_model'):
        obj = joblib.load(config[key])
        print(f'{key}: {config[key]}')
        print('keys:',sorted(obj.keys()))
        for k in ('input_mode','bounds','feature_names','targets','model_name','architecture',
                  'hidden_layer_sizes','n_train','support','training_bounds','sea_bounds'):
            if k in obj:
                print(k,repr(obj[k])[:3000])


def preflight(states, c, backend, root, ident, config_path):
    if backend == 'frozen':
        try:
            from pymoo.algorithms.moo.nsga2 import NSGA2
        except ImportError as exc:
            raise RuntimeError('Install pymoo==0.6.1.5 in the selected model environment before submitting the search') from exc
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    marker = root/'preflight.json'
    if marker.exists() and json.loads(marker.read_text())['fingerprint'] != ident['fingerprint']:
        raise RuntimeError('Output directory belongs to another code/config/model version; choose a new directory')
    atomic_json(root/'identity.json',ident)
    atomic_json(marker,{'fingerprint':ident['fingerprint'],'status':'RUNNING'})
    engine = make_backend(backend,c)
    rows = []
    actions = [np.zeros(10),np.array([-8,4,-3,6,1,.5,1.,0.,1.5,.5]),
               np.array([7,-6,2,-4,3,1.,.5,1.5,0.,1.])]
    comparisons = []
    for state in states:
        for j,action in enumerate(actions):
            start = time.perf_counter()
            fast = engine.evaluate(state,action)
            seconds = time.perf_counter()-start
            rows.append({'case':state.key,'action_index':j,'fast_seconds':seconds,
                         'power_kw':float(fast['power'].sum()),'status':'PASS'})
            if backend == 'analytic':
                continue
            # Compare the in-process API against the same frozen stack through the CLI.
            path = root/'parity'/f'{state.key}_action{j}.csv'
            path.parent.mkdir(exist_ok=True)
            cmd = [sys.executable,c['runner'],'--base-model',c['base_model'],'--sea-model',c['sea_model'],
                '--bundle',c['bundle'],'--pywake-script',str(Path(c['main'])/'pywake/bin/run_case.sh'),
                '--farm',state.farm,
                '--ws',str(state.ws),'--wd',str(state.wd),'--ti',str(state.ti),
                '--hs',str(state.hs),'--tp',str(state.tp),
                '--yaw='+','.join(map(str,action[:5])),
                '--pitch='+','.join(map(str,action[5:])), '--output',str(path.resolve())]
            p = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False)
            path.with_suffix('.log').write_text(p.stdout)
            if p.returncode:
                raise RuntimeError(f'Current runner failed; see {path.with_suffix(".log")}')
            reference = pd.read_csv(path).sort_values('Turbine')
            aerodynamic = pd.read_csv(path.with_suffix('.pywake.csv')).sort_values('Turbine')
            if reference.Turbine.tolist() != [1,2,3,4,5] or aerodynamic.Turbine.tolist() != [1,2,3,4,5]:
                raise RuntimeError('Reference turbine labels/order are not 1..5')
            target_cols = [col for col in reference.columns if col.startswith('sea_') and not col.startswith('sea_factor_')]
            order = target_order([x[4:] for x in target_cols])
            tests = {'loads':reference[[target_cols[k] for k in order]].to_numpy(float),
                'ws':reference.local_ws_mps.to_numpy(float),'ti':reference.local_ti_pct.to_numpy(float),
                'power':aerodynamic.Power_kW.to_numpy(float)}
            for field,ref in tests.items():
                if not np.allclose(fast[field],ref,rtol=c['parity_rtol'],atol=c['parity_atol']):
                    raise RuntimeError(f'{state.key} action {j}: {field} mismatch against current runner; do not optimize')
                comparisons.append({'case':state.key,'action_index':j,'field':field,
                    'max_absolute_error':float(np.max(np.abs(fast[field]-ref)))})
            print(f'Parity PASS: {state.key}, action {j}',flush=True)
    write_rows(root/'preflight_timings.csv',rows)
    write_rows(root/'preflight_parity.csv',comparisons)
    atomic_json(marker,{'fingerprint':ident['fingerprint'],'status':'PASS','backend':backend,
        'synthetic':backend=='analytic','n_test_actions':len(rows),
        'scope':'In-process frozen evaluator compared with the packaged CLI using the same assets',
        'config_path':str(Path(config_path).resolve()),
        'median_fast_seconds':float(np.median([r['fast_seconds'] for r in rows]))})
    label = 'synthetic action checks' if backend == 'analytic' else 'in-process/CLI comparisons'
    print(f'Preflight PASS: {len(rows)} {label}',flush=True)
