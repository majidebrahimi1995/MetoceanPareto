# MetoceanPareto

**Frozen five-turbine PyWake, floating-wind DEL surrogate, and exact-centroid Pareto set-point search.** This is the model and workflow package. Paper tables, figures, and FAST.Farm result archives belong in a separate results repository.

The package reflects the calibrated 3D OpenFAST power/thrust lookup, October 2026 base and sea-state load models, and C2 physical wind direction **167.83°**. The power objective is PyWake rotor power; FAST.Farm generator and rotor power are separate validation quantities.

## What is here

| Location | Purpose |
| --- | --- |
| `platform.py` | Single command entry point and frozen-asset checks |
| `deployment/CURRENT/pywake/` | Final aerodynamic calibration JSON, 3D OpenFAST table, and tilt correction table |
| `deployment/CURRENT/runtime/` | Exact frozen PyWake runtime dependencies |
| `pywake/bin/run_frozen_pywake_case.py` | Isolated aerodynamic calculation and optional velocity contour |
| `models/october_20261013/` | Frozen `base.joblib`, `sea.joblib`, and historical training manifest |
| `surrogate/smooth_del_v1/` | Connected PyWake → base/sea DEL evaluator |
| `optimization_pareto/centroid_pilot_v2/` | Three-restart, two-formulation exact-centroid optimizer and 0–10% budget extraction |
| `optimization_pareto/simple_cross_validation/` | Connected versus direct FAST.Farm inflow validation; no model fitting |
| `config/validation_environment.json` | Realized validation inflow; distinct from optimizer centroid inputs |
| `pywake/calibration/` | Optional refit source, requiring external calibration records and incumbent JSON |
| `scripts/submit_sge.sh` | Optional SGE job dependency chain |

There is no hidden dependency on `/global-scratch/...`. The historical provenance fields inside the unmodified calibration and model manifests still describe where the original models were trained. Those paths are **metadata**, not input paths for frozen inference.

## Install

Use Linux with Python **3.11**. The SGE workflow also needs `qsub` and an available `smp` parallel environment.

```bash
cd /path/to/MetoceanPareto
conda env create -f environment.yml
conda activate metocean-pareto
python platform.py doctor
```

Alternatively, create a Python 3.11 virtual environment, activate it, and run `python -m pip install -r requirements.txt`. The versions record the original cluster environment; other versions require a new parity check. The provided environment has **not** yet been installed and exercised in this workspace. `doctor` verifies installed versions, model files, and input table checksums.

Do not load `.joblib` assets from an untrusted source. This package contains the frozen models from the reviewed project snapshot.

## Run calibrated PyWake

```bash
python platform.py wake --farm Tampen --ws 8.2 --wd 167.83 --ti 4 \
  --yaw=-7.06194935,-16.18994912,-15.3481939,-2.642476142,-6.270880131 \
  --pitch=1.151400906,1.331883653,2.847901103,1.506098823,1.235849132 \
  --output results/Tampen_C2_aero.csv \
  --flow-map results/Tampen_C2_velocity.png --flow-map-resolution 600
```

This command should give about **30.837652 MW** farm rotor power in the recorded cluster environment. The flow-map command also writes a vector PDF. Inputs `--wd` and `--ti` are physical meteorological direction in degrees and ambient TI in percent. The 3D table is parameterized by wind speed, pitch, and yaw; `pitch_ct_tilt_map.csv` supplies the platform tilt correction.

## Run the connected load model

```bash
python platform.py loads --farm Tampen --ws 8.2 --wd 167.83 --ti 4 \
  --hs 2.804 --tp 11.014 \
  --yaw=0,0,0,0,0 --pitch=0,0,0,0,0 \
  --output results/Tampen_C2_loads.csv
```

The runner checks the hashes of the October base and sea models. It outputs turbine-level wake inflow and four DEL predictions plus a metadata JSON. For direct-load assessment, pass `--inflow-csv /path/to/FASTFarm_local_inflow.csv` with `Turbine,local_ws_mps,local_ti_pct`. This bypasses PyWake for load inputs; it does not change the load model. Predictions outside the trained action/sea domain are rejected by default.

## Run exact-centroid optimization

The frozen centroid file is `optimization_pareto/centroid_pilot_v2/config/centroids_validation_exact_wd167p830.csv`. C2 is 167.83°, and the other wind directions retain their exact recorded precision. The optimizer uses the centroid's own WS/TI; the validation environment JSON contains *realized FAST.Farm* ambient inputs and must not be substituted for it.

```bash
python platform.py pareto --output results/exact_centroids_v1 --workers 4
```

This runs preflight, 30 search tasks (10 layout–centroid conditions × 3 seeds), 10 cached merges, direction scan and report, 10 dense-budget extractions, and the combined dense report. It is computationally expensive. `--workers` controls CPU processes **per** search task. Completed tasks can be resumed using the same directory if code, models, inputs, and environment still match. A changed fingerprint requires a new output directory. The final selection database is `results/exact_centroids_v1/dense_budget_0to10/all_selected.csv`.

For SGE, activate the environment and supply the *actual queue name* and a new output directory:

```bash
PLATFORM_PYTHON="$(command -v python)" bash scripts/submit_sge.sh \
  meccsys1.q@node-m-9.local results/exact_centroids_v1 \
  optimization_pareto/centroid_pilot_v2/config/centroids_validation_exact_wd167p830.csv \
  16 4
```

This requests up to four 16-slot search tasks (64 slots total). Check queue capacity and wall-time policy before submission. The scripts send jobs to the queue you pass; no queue or Python path is hardcoded. Use `qstat` to monitor. For manual/resumable stage control, `python platform.py pareto --help` lists `--stage`, `--task-id`, and `--case`.

## Validate against new FAST.Farm results

Supply a seed-averaged FAST.Farm table with five turbine rows per farm, centroid, and strategy, plus the **matching** optimizer selection file:

```bash
python platform.py validate \
  --input /path/to/validation_characteristics_and_DEL_seedavg5.csv \
  --setpoints results/exact_centroids_v1/dense_budget_0to10/all_selected.csv \
  --environment config/validation_environment.json \
  --output-dir results/validation_v1 --workers 4 --audit-only

python platform.py validate \
  --input /path/to/validation_characteristics_and_DEL_seedavg5.csv \
  --setpoints results/exact_centroids_v1/dense_budget_0to10/all_selected.csv \
  --environment config/validation_environment.json \
  --output-dir results/validation_v1 --workers 4
```

First inspect the audit count, strategy list, and `direct_ti_clipped_rows`. Update the environment JSON when the realized FAST.Farm inflow changes; do not change the optimizer centroids to match realized values. The full run writes `predictions_ALL_5T.csv`, `farm_power_ALL_5T.csv`, `metrics_grouped_ALL_5T.csv`, `loads_direct_FASTFarm_inputs_ALL_5T.csv`, and `summary_ALL_5T.json`. Its connected branch uses PyWake local WS/TI; the direct branch uses FAST.Farm local WS and **saturates local TI to 4–14%** before the same DEL model. Interpret direct-branch errors with that saturation in mind. The input and setpoints are explicit, so rerunning after new simulations cannot silently select an older result folder.

## Optional PyWake recalibration

The exact final calibration refit source is included, but two original inputs are **not** in this snapshot: `absolute3d_full_wfm_1176/00_filtered_Mild1176.csv` and the `absolute3d_projectedCT_refit_seeded_40/pywake_absolute3d_calibration.json` incumbent. Obtain them from the project cluster before reproducing the 1,176-row fit. The frozen package works without them.

```bash
python pywake/calibration/openfast_turbine_calibration/refit_absolute3d_seeded_all_data_corrected.py \
  --main "$PWD/pywake" \
  --data /path/to/00_filtered_Mild1176.csv \
  --table3d "$PWD/deployment/CURRENT/pywake/openfast_absolute_3d_pywake_table_TI6_PROJECTED_CT_BOUNDED.csv" \
  --old-pitch-table "$PWD/deployment/CURRENT/pywake/pitch_ct_tilt_map.csv" \
  --incumbent-json /path/to/seeded_40/pywake_absolute3d_calibration.json \
  --output-dir calibration_runs/refit_01 --workers 4
```

The refit writes into a separate directory. It does not overwrite the frozen release. Releasing a new calibration requires a new asset manifest and full wake/connected parity verification. Retraining the DEL surrogate requires its original OpenFAST datasets and training pipeline, which are not in this frozen-model package.

## Release boundary

This archive is a **reviewable repository candidate**, not a published GitHub repository. Source compilation, checksum checks, synthetic preflight, and FAST.Farm input auditing have been exercised here. A real PyWake run and full optimizer run still need the recorded Python 3.11 environment on the cluster; the independent original-cluster parity result remains to be checked. Choose a license and review the historical provenance paths before public release. Keep paper figure generation and raw FAST.Farm validation campaigns in the separate results repository.
