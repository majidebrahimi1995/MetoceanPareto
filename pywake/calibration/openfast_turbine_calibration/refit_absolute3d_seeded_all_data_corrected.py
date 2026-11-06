#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# Explicit final search bounds. Do not inherit wake_bounds() from the legacy
# calibration module because that file may be configured for a different test.
WAKE_BOUNDS = [(0.015, 0.120), (0.15, 0.60), (0.0, 1.50)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--table3d', type=Path, required=True)
    ap.add_argument('--old-pitch-table', type=Path, required=True)
    ap.add_argument('--incumbent-json', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('NSLOTS','1')))
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--maxiter-ws', type=int, default=90)
    ap.add_argument('--popsize-ws', type=int, default=30)
    ap.add_argument('--maxiter-ti', type=int, default=90)
    ap.add_argument('--popsize-ti', type=int, default=44)
    a = ap.parse_args()

    main_root = a.main.resolve()
    data_path = a.data.resolve()
    table3d = a.table3d.resolve()
    old_pitch = a.old_pitch_table.resolve()
    incumbent_json = a.incumbent_json.resolve()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    for p in [data_path, table3d, old_pitch, incumbent_json]:
        if not p.is_file():
            raise FileNotFoundError(p)

    cal = main_root/'calibration'
    root = cal/'openfast_turbine_calibration'
    sys.path.insert(0, str(root)); sys.path.insert(0, str(cal)); sys.path.insert(0, str(main_root/'src'))

    import calibrate_pywake_t2 as c
    from openfast_absolute_3d_turbine import build_absolute_openfast_3d_turbine

    print('='*100, flush=True)
    print('ABSOLUTE OPENFAST 3-D SEEDED ALL-DATA REFIT', flush=True)
    print('='*100, flush=True)
    print('Python          :', sys.executable, flush=True)
    print('Data            :', data_path, flush=True)
    print('Absolute 3-D    :', table3d, flush=True)
    print('Tilt table only :', old_pitch, flush=True)
    print('Incumbent JSON  :', incumbent_json, flush=True)
    print('Workers         :', a.workers, flush=True)
    print('Output          :', out, flush=True)

    data, mapping, split_keys = c.load_and_prepare_data(
        data_path, average_sea_state=False, wd_mode='fixed', fixed_wd=270.0,
        max_rows=None, random_seed=a.seed)
    arrays = c.dataframe_to_arrays(data)
    if len(data) != 1176:
        raise RuntimeError(f'Expected 1176 rows, got {len(data)}')

    original_builder = c.build_pitch_table_model_cached
    turbine3d, info3d = build_absolute_openfast_3d_turbine(table3d, diameter_m=240.0, hub_height_m=150.0)
    old_model = original_builder(str(old_pitch), 240.0, 150.0, 1.225)
    composite = c.PitchTableModel(
        turbine=turbine3d,
        table=old_model.table,
        ws_grid=np.asarray(old_model.ws_grid,float),
        pitch_grid=np.asarray(old_model.pitch_grid,float),
        platform_pitch_table=old_model.platform_pitch_table)

    def patched_builder(*args, **kwargs):
        return composite
    c.build_pitch_table_model_cached = patched_builder

    spec = c.CandidateSpec(
        model='zong', rotor_model='gaussian_overlap', yaw_sign=1,
        tilt_mode='six_plus_csv', fixed_tilt_deg=6.0, tilt_iterations=2,
        diameter=240.0, hub_height=150.0, rho=1.225,
        pitch_table=str(old_pitch))

    inc = json.loads(incumbent_json.read_text(encoding='utf-8'))
    wake_inc = np.asarray(inc['wake_parameters'], float)
    crespo_full_inc = np.asarray(inc['crespo_c'], float)
    crespo_inc = np.array([crespo_full_inc[0], crespo_full_inc[3]], float)

    print('\nIncumbent wake  :', wake_inc.tolist(), flush=True)
    print('Incumbent Crespo:', crespo_full_inc.tolist(), flush=True)
    print('3-D WS grid     :', np.asarray(info3d['ws_grid']).tolist(), flush=True)
    print('3-D pitch grid  :', np.asarray(info3d['pitch_grid']).tolist(), flush=True)
    print('3-D yaw grid    :', np.asarray(info3d['yaw_grid']).tolist(), flush=True)

    ws0, ti0 = c.predict_one(arrays, 0, spec, wake_inc, float(crespo_inc[0]), float(crespo_inc[1]))
    print(f'Smoke row 0 -> T2 WS={ws0:.6f} m/s, TI={ti0:.6f} %', flush=True)

    ws_obj = c.WSObjective(arrays, spec)
    ws_inc_obj = float(ws_obj(wake_inc))
    if not np.isfinite(ws_inc_obj) or ws_inc_obj >= 1e8:
        raise RuntimeError(f'Invalid incumbent WS objective: {ws_inc_obj}')
    print('\n'+'='*100, flush=True); print('WS REFIT', flush=True); print('='*100, flush=True)
    print('Incumbent WS objective:', ws_inc_obj, flush=True)
    print('Explicit WS bounds     :', WAKE_BOUNDS, flush=True)
    for j, (x, (lo, hi)) in enumerate(zip(wake_inc, WAKE_BOUNDS)):
        if not (lo <= float(x) <= hi):
            raise RuntimeError(f'Incumbent wake parameter {j}={x} is outside explicit bounds {lo}..{hi}')

    t0=time.perf_counter()
    ws_res = differential_evolution(
        ws_obj, bounds=WAKE_BOUNDS, x0=wake_inc,
        maxiter=a.maxiter_ws, popsize=a.popsize_ws,
        tol=1e-7, atol=0.0, mutation=(0.5,1.0), recombination=0.7,
        seed=a.seed, polish=True, workers=max(1,a.workers),
        updating='deferred' if a.workers>1 else 'immediate',
        init='latinhypercube', disp=True)
    ws_seconds=time.perf_counter()-t0
    wake_candidate=np.asarray(ws_res.x,float); ws_candidate_obj=float(ws_res.fun)
    if ws_candidate_obj <= ws_inc_obj + 1e-15:
        wake_final=wake_candidate; ws_final_obj=ws_candidate_obj; ws_guard='optimizer_candidate'
    else:
        wake_final=wake_inc.copy(); ws_final_obj=ws_inc_obj; ws_guard='incumbent_retained'
    print('\nWS optimizer candidate:', wake_candidate.tolist(), flush=True)
    print('WS candidate objective :', ws_candidate_obj, flush=True)
    print('WS selected            :', wake_final.tolist(), flush=True)
    print('WS selected objective  :', ws_final_obj, flush=True)
    print('WS incumbent guard     :', ws_guard, flush=True)

    ti_obj = c.TIObjective(arrays, spec, wake_final)
    ti_inc_obj=float(ti_obj(crespo_inc))
    if not np.isfinite(ti_inc_obj) or ti_inc_obj >= 1e8:
        raise RuntimeError(f'Invalid incumbent TI objective: {ti_inc_obj}')
    ti_bounds=[(0.10,6.00),(-2.00,-0.05)]
    print('\n'+'='*100, flush=True); print('TI REFIT', flush=True); print('='*100, flush=True)
    print('Incumbent TI objective:', ti_inc_obj, flush=True)

    wake_changed = not np.allclose(wake_final, wake_inc, rtol=0.0, atol=1e-14)
    if not wake_changed:
        # The supplied incumbent JSON already contains a converged TI refit for
        # exactly this wake solution and exactly this absolute-3D turbine model.
        # Reuse it instead of spending ~4 h repeating an identical optimization.
        c_candidate = crespo_inc.copy()
        ti_candidate_obj = ti_inc_obj
        c_final = crespo_inc.copy()
        ti_final_obj = ti_inc_obj
        ti_guard = 'reused_incumbent_same_wake'
        ti_seconds = 0.0
        ti_res = None
        print('Wake unchanged: reusing incumbent Crespo; TI DE skipped.', flush=True)
    else:
        t1=time.perf_counter()
        ti_res=differential_evolution(
            ti_obj, bounds=ti_bounds, x0=crespo_inc,
            maxiter=a.maxiter_ti, popsize=a.popsize_ti,
            tol=1e-7, atol=0.0, mutation=(0.5,1.0), recombination=0.7,
            seed=a.seed+1000, polish=True, workers=max(1,a.workers),
            updating='deferred' if a.workers>1 else 'immediate',
            init='latinhypercube', disp=True)
        ti_seconds=time.perf_counter()-t1
        c_candidate=np.asarray(ti_res.x,float); ti_candidate_obj=float(ti_res.fun)
        if ti_candidate_obj <= ti_inc_obj + 1e-15:
            c_final=c_candidate; ti_final_obj=ti_candidate_obj; ti_guard='optimizer_candidate'
        else:
            c_final=crespo_inc.copy(); ti_final_obj=ti_inc_obj; ti_guard='incumbent_retained'
    crespo_final=[float(c_final[0]),0.8325,0.0325,float(c_final[1])]
    print('\nTI optimizer candidate:', c_candidate.tolist(), flush=True)
    print('TI candidate objective :', ti_candidate_obj, flush=True)
    print('TI selected Crespo     :', crespo_final, flush=True)
    print('TI selected objective  :', ti_final_obj, flush=True)
    print('TI incumbent guard     :', ti_guard, flush=True)

    print('\nComputing final 1176 predictions...', flush=True)
    pred_ws, pred_ti = c.predict_dataset(arrays, spec, wake_final, float(c_final[0]), float(c_final[1]))
    ws_metrics=c.ws_metrics(arrays['t2_ws'], pred_ws)
    ti_metrics=c.ti_metrics(arrays['t2_ti_pct'], pred_ti)
    print('\nFINAL WS METRICS'); [print(f'  {k}: {v}') for k,v in ws_metrics.items()]
    print('\nFINAL TI METRICS'); [print(f'  {k}: {v}') for k,v in ti_metrics.items()]

    pred=data.copy()
    pred['pywake_t2_ws_mps']=pred_ws; pred['pywake_t2_ti_pct']=pred_ti
    pred['ws_error_mps']=pred_ws-arrays['t2_ws']
    pred['ws_error_pct']=100*pred['ws_error_mps']/np.maximum(np.abs(arrays['t2_ws']),1e-9)
    pred['ti_error_percentage_points']=pred_ti-arrays['t2_ti_pct']
    pred.to_csv(out/'calibrated_predictions_all.csv', index=False)

    cfg={
        'schema':'absolute_openfast_3d_seeded_all_data_refit_v1',
        'all_data_refit':True,
        'optimization_variant':'seeded_incumbent_guard_absolute3d',
        'number_of_rows':int(len(data)),
        'number_of_physical_groups':int(data['physical_group'].nunique()),
        'wake_model':'zong','rotor_average_model':'gaussian_overlap','yaw_sign':1,
        'tilt_mode':'six_plus_csv','fixed_tilt_deg':6.0,'tilt_iterations':2,
        'diameter_m':240.0,'hub_height_m':150.0,'rho_kg_m3':1.225,
        'absolute_3d_table':str(table3d),'absolute_3d_table_sha256':sha256(table3d),
        'absolute_3d_inputs':['ws','pitch_offset_deg','yaw'],
        'absolute_3d_power':'OpenFAST mean_RotPwr',
        'absolute_3d_ct_for_wake':'OpenFAST mean_RtFldCt * cos(yaw_command)^2',
        'simple_yaw_model':False,'density_scale':True,
        'platform_tilt_table':str(old_pitch),'platform_tilt_table_sha256':sha256(old_pitch),
        'data':str(data_path),'data_sha256':sha256(data_path),
        'column_mapping':mapping,'physical_split_keys':split_keys,
        'incumbent_json':str(incumbent_json),'incumbent_json_sha256':sha256(incumbent_json),
        'incumbent_wake_parameters':wake_inc.tolist(),'incumbent_crespo_c':crespo_full_inc.tolist(),
        'wake_parameter_names':c.parameter_names('zong'),
        'wake_parameters':wake_final.tolist(),
        'wake_parameter_dict':dict(zip(c.parameter_names('zong'),wake_final.tolist())),
        'crespo_c':crespo_final,
        'ws_objective_incumbent':ws_inc_obj,'ws_objective_optimizer_candidate':ws_candidate_obj,
        'ws_optimizer_candidate_parameters':wake_candidate.tolist(),
        'ws_objective_final':ws_final_obj,'ws_incumbent_guard':ws_guard,
        'ti_objective_incumbent':ti_inc_obj,'ti_objective_optimizer_candidate':ti_candidate_obj,
        'ti_optimizer_candidate_c0_c3':c_candidate.tolist(),
        'ti_objective_final':ti_final_obj,'ti_incumbent_guard':ti_guard,
        'all_data_ws_metrics':ws_metrics,'all_data_ti_metrics':ti_metrics,
        'optimizer':{
            'workers':int(a.workers),'seed':int(a.seed),
            'ws':{'bounds':WAKE_BOUNDS,'maxiter':a.maxiter_ws,'popsize':a.popsize_ws,
                  'tol':1e-7,'atol':0.0,'nfev':int(ws_res.nfev),'success':bool(ws_res.success),
                  'message':str(ws_res.message),'seconds':float(ws_seconds)},
            'ti':{'bounds':ti_bounds,'maxiter':a.maxiter_ti,'popsize':a.popsize_ti,
                  'tol':1e-7,'atol':0.0,
                  'nfev':0 if ti_res is None else int(ti_res.nfev),
                  'success':True if ti_res is None else bool(ti_res.success),
                  'message':'reused incumbent because wake was unchanged' if ti_res is None else str(ti_res.message),
                  'seconds':float(ti_seconds)}
        }
    }
    write_json(out/'pywake_absolute3d_calibration.json', cfg)
    (out/'FINAL_SUCCESS').write_text('success\n',encoding='utf-8')
    print('\n'+'='*100)
    print('FINAL SELECTED MODEL')
    print('='*100)
    print('Wake   :', wake_final.tolist())
    print('Crespo :', crespo_final)
    print('WS obj :', ws_final_obj)
    print('TI obj :', ti_final_obj)
    print('Output :', out)
    print('FINAL_SUCCESS written.')

if __name__=='__main__':
    main()
