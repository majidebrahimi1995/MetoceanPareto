#!/usr/bin/env python3
"""One-command 5-turbine validation using the current October DEL models.

Inputs: explicit FAST.Farm seed-averaged CSV and exact optimizer all_selected.csv.

Outputs:
  predictions_ALL_5T.csv
  farm_power_ALL_5T.csv
  metrics_grouped_ALL_5T.csv
  loads_direct_FASTFarm_inputs_ALL_5T.csv
  summary_ALL_5T.json

Two load predictions are produced with the exact same base+sea models:
  CONNECTED: ambient conditions -> PyWake -> local WS/TI -> DEL model
  DIRECT:    FAST.Farm local WS/TI              -> DEL model
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


MAIN = Path(__file__).resolve().parents[2]
PILOT = MAIN / "optimization_pareto/centroid_pilot_v2"
BUNDLE = MAIN / "surrogate/smooth_del_v1"
REFIT = MAIN / "models/october_20261013"
PYTHON = Path(sys.executable)
RUNNER = BUNDLE / "run_pywake_sea_surrogate.py"
FROZEN_PYWAKE_RUNNER = MAIN / "pywake/bin/run_frozen_pywake_case.py"
PYWAKE_STACK = MAIN / "deployment/CURRENT"
BASE_MODEL = REFIT / "base.joblib"
SEA_MODEL = REFIT / "sea.joblib"

EXPECTED_HASHES = {
    "base.joblib": "5260b2f50857f53f60a83faae56a210a251e3f51d9b684ab2456421c341ae679",
    "sea.joblib": "2f57f175b33358c58fa03eb480b3b4165a66a5c2d8ba174971c00ef8f55c9676",
    "surrogate_core.py": "97781ba6442a3b2ef696d85c5f74aa8039c881ce8a730195ef89177cccfe3c32",
}

VALIDATION_ENV = {}

STRATEGY_TO_OPT = {
    "GREEDY": "Greedy",
    "MAXPWR": "Pbest",
    "STR03": "STR3_tau1.025",
    "STR06": "STR6_tau1.025",
    "MOR03": "MOR3_tau1.025",
    "MOR06": "MOR6_tau1.025",
    "COM03": "BAL3_tau1.025",
    "COM06": "BAL6_tau1.025",
}

DEL_CHANNELS = {
    "TowerBase": ("DEL_TowerBase_Max_NoGoodman", "sea_DEL_TowerBase"),
    "BladeRoot": ("DEL_BladeRoot_Max_FixedMean", "sea_DEL_BladeRoot"),
    "YawBearing": ("DEL_YawBearing_Max_NoGoodman", "sea_DEL_YawBearing"),
    "FairTenMax": ("FASTFarm_DEL_FairTenMax", "sea_DEL_FairTenMax"),
}

REQUIRED_FASTFARM = {
    "Farm", "Turbine", "MetoceanCase", "StrategyTag",
    "Mean_RotorEffectiveSpeed_mps", "TI_RotorEffective_pct",
    "Mean_GenPwr", "Mean_RotPwr",
    "DEL_TowerBase_Max_NoGoodman", "DEL_BladeRoot_Max_FixedMean",
    "DEL_YawBearing_Max_NoGoodman", "DEL_Fairlead1_NoGoodman",
    "DEL_Fairlead2_NoGoodman", "DEL_Fairlead3_NoGoodman",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_farm(value: object) -> str:
    text = str(value).strip().lower()
    if text == "rectangle":
        return "Rectangle"
    if text == "tampen":
        return "Tampen"
    raise ValueError(f"Unsupported farm: {value!r}")


def normalize_strategy(value: object) -> str:
    text = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "GREEDY": "GREEDY", "PBEST": "MAXPWR", "MAXPWR": "MAXPWR",
        "STR3": "STR03", "STR03": "STR03", "STR6": "STR06", "STR06": "STR06",
        "MOR3": "MOR03", "MOR03": "MOR03", "MOR6": "MOR06", "MOR06": "MOR06",
        "BAL3": "COM03", "BAL03": "COM03", "COM3": "COM03", "COM03": "COM03",
        "BAL6": "COM06", "BAL06": "COM06", "COM6": "COM06", "COM06": "COM06",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported StrategyTag: {value!r}")
    return aliases[text]


def locate_fastfarm_csv(folder: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = []
    for path in folder.glob("*.csv"):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if REQUIRED_FASTFARM.issubset(columns):
            candidates.append(path.resolve())
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(
            "Place exactly one FAST.Farm validation CSV beside run.sh, or pass "
            f"--input. Matching files: {names}"
        )
    return candidates[0]


def c2_is_correct(path: Path) -> bool:
    try:
        frame = pd.read_csv(path, usecols=["centroid", "wd"])
        wd = pd.to_numeric(frame.loc[frame["centroid"].astype(str).str.upper() == "C2", "wd"], errors="coerce")
        return len(wd) > 0 and np.allclose(wd.dropna().to_numpy(), 167.83, atol=1e-8)
    except Exception:
        return False


def locate_setpoints(explicit: Path | None) -> Path:
    if explicit is None:
        raise ValueError("Pass --setpoints with the exact optimizer all_selected.csv")
    path = explicit.resolve()
    if not path.is_file() or not c2_is_correct(path):
        raise RuntimeError(f"Setpoint CSV must exist and contain C2 WD=167.83: {path}")
    return path


def audit_model_stack() -> dict[str, str]:
    paths = {
        "base.joblib": BASE_MODEL,
        "sea.joblib": SEA_MODEL,
        "surrogate_core.py": BUNDLE / "surrogate_core.py",
    }
    actual = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual[name] = sha256(path)
        if actual[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(
                f"Wrong model file {path}\nexpected {EXPECTED_HASHES[name]}\nfound    {actual[name]}"
            )
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    if not FROZEN_PYWAKE_RUNNER.is_file():
        raise FileNotFoundError(FROZEN_PYWAKE_RUNNER)
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    return actual


def load_fastfarm(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_FASTFARM - set(frame.columns))
    if missing:
        raise ValueError(f"FAST.Farm CSV is missing columns: {missing}")
    frame = frame.copy()
    frame["Farm"] = frame["Farm"].map(normalize_farm)
    frame["StrategyTag"] = frame["StrategyTag"].map(normalize_strategy)
    frame["MetoceanCase"] = pd.to_numeric(frame["MetoceanCase"], errors="raise").astype(int)
    frame["Turbine"] = pd.to_numeric(frame["Turbine"], errors="raise").astype(int)
    invalid_case = sorted(set(frame["MetoceanCase"]) - set(VALIDATION_ENV))
    if invalid_case:
        raise ValueError(f"Unsupported metocean cases: {invalid_case}")
    invalid_turbine = sorted(set(frame["Turbine"]) - {1, 2, 3, 4, 5})
    if invalid_turbine:
        raise ValueError(f"Turbines must be 1..5; found {invalid_turbine}")
    fairlead = ["DEL_Fairlead1_NoGoodman", "DEL_Fairlead2_NoGoodman", "DEL_Fairlead3_NoGoodman"]
    frame["FASTFarm_DEL_FairTenMax"] = frame[fairlead].max(axis=1)
    keys = ["Farm", "MetoceanCase", "StrategyTag", "Turbine"]
    if frame.duplicated(keys).any():
        raise ValueError(
            "Input must be seed-averaged: duplicate Farm/Case/Strategy/Turbine rows were found."
        )
    counts = frame.groupby(["Farm", "MetoceanCase", "StrategyTag"])["Turbine"].nunique()
    bad = counts[counts != 5]
    if len(bad):
        raise ValueError(f"Every simulation block must contain five turbines:\n{bad}")
    return frame.sort_values(keys).reset_index(drop=True)


def build_setpoint_lookup(path: Path, wanted: set[tuple[str, int, str]]) -> dict[tuple[str, int, str], dict]:
    source = pd.read_csv(path)
    required = {"farm", "centroid", "wd", "strategy", "power_kw"}
    required |= {f"yaw_T{i}" for i in range(1, 6)}
    required |= {f"pitch_T{i}" for i in range(1, 6)}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"all_selected.csv is missing columns: {missing}")
    lookup = {}
    for farm, case, strategy in sorted(wanted):
        optimizer_strategy = STRATEGY_TO_OPT[strategy]
        mask = (
            source["farm"].astype(str).str.casefold().eq(farm.casefold())
            & source["centroid"].astype(str).str.upper().eq(f"C{case}")
            & source["strategy"].astype(str).eq(optimizer_strategy)
        )
        selected = source.loc[mask]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one optimizer row for {farm}/C{case}/{optimizer_strategy}; found {len(selected)}"
            )
        row = selected.iloc[0]
        expected_wd = VALIDATION_ENV[case]["wd"]
        if not math.isclose(float(row["wd"]), expected_wd, abs_tol=1e-8):
            raise RuntimeError(
                f"Wrong WD for {farm}/C{case}: {row['wd']} (expected {expected_wd})"
            )
        lookup[(farm, case, strategy)] = {
            "yaw": [float(row[f"yaw_T{i}"]) for i in range(1, 6)],
            "pitch": [float(row[f"pitch_T{i}"]) for i in range(1, 6)],
            "power_kw": float(row["power_kw"]),
            "optimizer_strategy": optimizer_strategy,
        }
    return lookup


def csv_vector(values: list[float]) -> str:
    return ",".join(f"{value:.12g}" for value in values)


def run_case(*, farm: str, case: int, strategy: str, commands: dict,
             direct_inflow: pd.DataFrame | None, output: Path) -> pd.DataFrame:
    env = VALIDATION_ENV[case]
    command = [
        str(PYTHON), str(RUNNER),
        "--base-model", str(BASE_MODEL),
        "--sea-model", str(SEA_MODEL),
                "--farm", farm,
        "--ws", str(env["ws"]),
        "--wd", str(env["wd"]),
        "--ti", str(env["ti"]),
        "--hs", str(env["hs"]),
        "--tp", str(env["tp"]),
        f"--yaw={csv_vector(commands['yaw'])}",
        f"--pitch={csv_vector(commands['pitch'])}",
        "--output", str(output),
    ]
    if direct_inflow is not None:
        inflow_path = output.with_suffix(".inflow.csv")
        bounded_inflow = direct_inflow[["Turbine", "local_ws_mps", "local_ti_pct"]].copy()
        bounded_inflow["local_ti_pct"] = bounded_inflow["local_ti_pct"].clip(lower=4.0, upper=14.0)
        bounded_inflow.to_csv(inflow_path, index=False)
        command.extend([
            "--inflow-csv", str(inflow_path),
            "--pywake-ti-units", "percent",
        ])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Model runner failed for {farm}/C{case}/{strategy}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    result = pd.read_csv(output)
    needed = {"Turbine", "local_ws_mps", "local_ti_pct"}
    needed |= {model for _, model in DEL_CHANNELS.values()}
    missing = sorted(needed - set(result.columns))
    if missing:
        raise RuntimeError(f"Runner output {output} is missing: {missing}")
    result["Farm"] = farm
    result["MetoceanCase"] = case
    result["StrategyTag"] = strategy
    return result


def run_power_case(*, farm: str, case: int, commands: dict,
                   output: Path) -> float:
    """Recompute farm power at the actual FAST.Farm validation condition."""
    env = VALIDATION_ENV[case]
    command = [
        str(PYTHON), str(FROZEN_PYWAKE_RUNNER),
        "--stack", str(PYWAKE_STACK),
        "--farm", farm,
        "--ws", str(env["ws"]),
        "--wd", str(env["wd"]),
        "--ti", str(env["ti"]),
        f"--yaw={csv_vector(commands['yaw'])}",
        f"--pitch={csv_vector(commands['pitch'])}",
        "--output", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Frozen PyWake power runner failed for {farm}/C{case}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    result = pd.read_csv(output)
    if "Power_kW" not in result.columns:
        raise RuntimeError(f"Power output {output} has no Power_kW column")
    if len(result) != 5:
        raise RuntimeError(
            f"Power output {output} contains {len(result)} rows; expected 5"
        )
    power = pd.to_numeric(result["Power_kW"], errors="raise").to_numpy(float)
    if not np.isfinite(power).all():
        raise RuntimeError(f"Power output {output} contains non-finite values")
    return float(power.sum())


def metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if len(actual) == 0:
        return {"N": 0, "MAE": math.nan, "RMSE": math.nan, "Bias": math.nan,
                "MAPE_pct": math.nan, "RMSPE_pct": math.nan, "R2": math.nan}
    error = predicted - actual
    nz = np.abs(actual) > 1e-12
    pct = 100.0 * error[nz] / actual[nz]
    denominator = np.sum((actual - actual.mean()) ** 2)
    r2 = math.nan if denominator <= 0 else 1.0 - float(np.sum(error ** 2) / denominator)
    return {
        "N": int(len(actual)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "Bias": float(np.mean(error)),
        "MAPE_pct": float(np.mean(np.abs(pct))) if len(pct) else math.nan,
        "RMSPE_pct": float(np.sqrt(np.mean(pct ** 2))) if len(pct) else math.nan,
        "R2": r2,
    }


def append_metrics(rows: list[dict], frame: pd.DataFrame, branch: str,
                   grouping: str, group_value: str) -> None:
    suffix = "Connected" if branch == "CONNECTED" else "Direct"
    for channel, (actual_col, model_col) in DEL_CHANNELS.items():
        predicted_col = f"{model_col}_{suffix}"
        values = metric(frame[actual_col].to_numpy(float), frame[predicted_col].to_numpy(float))
        rows.append({"Branch": branch, "Grouping": grouping, "Group": group_value,
                     "Channel": channel, **values})


def create_metrics(predictions: pd.DataFrame, farm_power: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for branch in ("CONNECTED", "DIRECT"):
        append_metrics(rows, predictions, branch, "Overall", "ALL")
        for farm, part in predictions.groupby("Farm", sort=True):
            append_metrics(rows, part, branch, "Farm", str(farm))
        for strategy, part in predictions.groupby("StrategyTag", sort=True):
            append_metrics(rows, part, branch, "Strategy", str(strategy))
        for case, part in predictions.groupby("MetoceanCase", sort=True):
            append_metrics(rows, part, branch, "MetoceanCase", f"C{case}")
        for (farm, strategy), part in predictions.groupby(["Farm", "StrategyTag"], sort=True):
            append_metrics(rows, part, branch, "Farm_Strategy", f"{farm}|{strategy}")
    power_values = metric(
        farm_power["FASTFarm_RotorPower_kW"].to_numpy(float),
        farm_power["PyWake_FarmPower_kW"].to_numpy(float),
    )
    rows.append({"Branch": "CONNECTED", "Grouping": "Overall", "Group": "ALL",
                 "Channel": "FarmPower", **power_values})
    for farm, part in farm_power.groupby("Farm", sort=True):
        values = metric(part["FASTFarm_RotorPower_kW"].to_numpy(float),
                        part["PyWake_FarmPower_kW"].to_numpy(float))
        rows.append({"Branch": "CONNECTED", "Grouping": "Farm", "Group": farm,
                     "Channel": "FarmPower", **values})
    return pd.DataFrame(rows)


def main() -> int:
    global VALIDATION_ENV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FAST.Farm seed-averaged CSV")
    parser.add_argument("--setpoints", type=Path, required=True, help="Exact optimizer all_selected.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", type=Path,
                        default=MAIN / "config/validation_environment.json",
                        help="Realized FAST.Farm ambient WS/TI/WD/Hs/Tp by case")
    parser.add_argument(
        "--exclude-cases", type=int, nargs="*", default=[], metavar="CASE",
        help="Optional metocean cases to omit from a diagnostic run",
    )
    parser.add_argument("--audit-only", action="store_true", help="Check inputs without model calls")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of validation blocks evaluated concurrently",
    )
    args = parser.parse_args()
    source = json.loads(args.environment.read_text())
    VALIDATION_ENV = {int(key): dict(value) for key, value in source.items()}
    if set(VALIDATION_ENV) != set(range(1, 6)) or any(
        set(v) != {'ws','ti','wd','hs','tp'} or not all(math.isfinite(float(x)) for x in v.values())
        for v in VALIDATION_ENV.values()
    ):
        raise ValueError('Environment JSON requires cases 1..5 with finite ws, ti, wd, hs, tp')
    if not math.isclose(float(VALIDATION_ENV[2]['wd']), 167.83, abs_tol=1e-8):
        raise ValueError('C2 validation wind direction must be 167.83 degrees')
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    folder = Path(__file__).resolve().parent
    input_path = locate_fastfarm_csv(folder, args.input)
    setpoint_path = locate_setpoints(args.setpoints)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_hashes = {} if args.audit_only else audit_model_stack()
    fastfarm = load_fastfarm(input_path)
    excluded_cases = sorted(set(args.exclude_cases))
    if excluded_cases:
        unknown = sorted(set(excluded_cases) - set(VALIDATION_ENV))
        if unknown:
            raise ValueError(f"Cannot exclude unknown metocean cases: {unknown}")
        before = len(fastfarm)
        fastfarm = fastfarm[~fastfarm["MetoceanCase"].isin(excluded_cases)].copy()
        if fastfarm.empty:
            raise ValueError("All input rows were excluded")
        print(f"Excluded cases {excluded_cases}: {before - len(fastfarm)} turbine rows omitted")
    wanted = set(map(tuple, fastfarm[["Farm", "MetoceanCase", "StrategyTag"]].drop_duplicates().to_numpy()))
    commands = build_setpoint_lookup(setpoint_path, wanted)

    audit = {
        "input_csv": str(input_path),
        "environment_json": str(args.environment.resolve()),
        "setpoint_csv": str(setpoint_path),
        "turbine_rows": int(len(fastfarm)),
        "simulation_blocks": int(len(wanted)),
        "direct_ti_policy": "clip FAST.Farm local TI to [4, 14] percent before load prediction",
        "direct_ti_clipped_rows": int(((fastfarm["TI_RotorEffective_pct"] < 4.0) |
                                        (fastfarm["TI_RotorEffective_pct"] > 14.0)).sum()),
        "farms": sorted(fastfarm["Farm"].unique().tolist()),
        "cases": sorted(map(int, fastfarm["MetoceanCase"].unique())),
        "strategies": sorted(fastfarm["StrategyTag"].unique().tolist()),
        "excluded_cases": excluded_cases,
        "correct_c2_wd_deg": 167.83,
        "workers": args.workers,
        "validation_environment": VALIDATION_ENV,
        "model_hashes": model_hashes,
    }
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        (output_dir / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        return 0

    task_list = sorted(
        (str(farm), int(case), str(strategy))
        for farm, case, strategy in wanted
    )
    results = {}

    with tempfile.TemporaryDirectory(prefix="five_turbine_validation_") as tmp:
        temporary = Path(tmp)

        def evaluate_block(task):
            farm, case, strategy = task
            key = (farm, int(case), strategy)
            block = fastfarm[
                (fastfarm["Farm"] == farm)
                & (fastfarm["MetoceanCase"] == case)
                & (fastfarm["StrategyTag"] == strategy)
            ].sort_values("Turbine")
            inflow = block[["Turbine", "Mean_RotorEffectiveSpeed_mps", "TI_RotorEffective_pct"]].rename(
                columns={"Mean_RotorEffectiveSpeed_mps": "local_ws_mps",
                         "TI_RotorEffective_pct": "local_ti_pct"}
            )
            slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{farm}_C{case}_{strategy}")
            connected_result = run_case(
                farm=farm, case=int(case), strategy=strategy, commands=commands[key],
                direct_inflow=None, output=temporary / f"{slug}_connected.csv")
            direct_result = run_case(
                farm=farm, case=int(case), strategy=strategy, commands=commands[key],
                direct_inflow=inflow, output=temporary / f"{slug}_direct.csv")
            fresh_power_kw = run_power_case(
                farm=farm, case=int(case), commands=commands[key],
                output=temporary / f"{slug}_power.csv")
            return key, connected_result, direct_result, fresh_power_kw

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_block, task): task
                for task in task_list
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    key, connected_result, direct_result, fresh_power_kw = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Validation failed for {task[0]}/C{task[1]}/{task[2]}"
                    ) from exc
                results[key] = (connected_result, direct_result, fresh_power_kw)
                print(
                    f"[{completed_count:02d}/{len(task_list):02d}] completed "
                    f"{key[0]} C{key[1]} {key[2]}",
                    flush=True,
                )

    connected_frames = [results[key][0] for key in task_list]
    direct_frames = [results[key][1] for key in task_list]
    pywake_power = {key: results[key][2] for key in task_list}

    join_keys = ["Farm", "MetoceanCase", "StrategyTag", "Turbine"]
    connected = pd.concat(connected_frames, ignore_index=True)
    direct = pd.concat(direct_frames, ignore_index=True)
    connected = connected.rename(columns={
        "local_ws_mps": "PyWake_local_ws_mps",
        "local_ti_pct": "PyWake_local_ti_pct",
        **{model: f"{model}_Connected" for _, model in DEL_CHANNELS.values()},
    })
    direct = direct.rename(columns={
        "local_ws_mps": "FASTFarmInput_local_ws_mps",
        "local_ti_pct": "FASTFarmInput_local_ti_pct",
        **{model: f"{model}_Direct" for _, model in DEL_CHANNELS.values()},
    })
    connected_keep = join_keys + ["PyWake_local_ws_mps", "PyWake_local_ti_pct"] + [
        f"{model}_Connected" for _, model in DEL_CHANNELS.values()]
    direct_keep = join_keys + ["FASTFarmInput_local_ws_mps", "FASTFarmInput_local_ti_pct"] + [
        f"{model}_Direct" for _, model in DEL_CHANNELS.values()]
    predictions = fastfarm.merge(connected[connected_keep], on=join_keys, validate="one_to_one")
    predictions = predictions.merge(direct[direct_keep], on=join_keys, validate="one_to_one")
    predictions["Direct_TI_WasClipped"] = ~np.isclose(
        predictions["TI_RotorEffective_pct"].to_numpy(float),
        predictions["FASTFarmInput_local_ti_pct"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )

    farm_rows = []
    for key, block in predictions.groupby(["Farm", "MetoceanCase", "StrategyTag"], sort=True):
        normalized_key = (str(key[0]), int(key[1]), str(key[2]))
        command = commands[normalized_key]
        recomputed_power = pywake_power[normalized_key]
        fastfarm_rotor_power = float(block["Mean_RotPwr"].sum())
        farm_rows.append({
            "Farm": key[0], "MetoceanCase": int(key[1]), "StrategyTag": key[2],
            "FASTFarm_GenPower_kW": float(block["Mean_GenPwr"].sum()),
            "FASTFarm_RotorPower_kW": fastfarm_rotor_power,
            "PyWake_FarmPower_kW": recomputed_power,
            "OptimizerStoredPower_kW": command["power_kw"],
            "OptimizerStrategy": command["optimizer_strategy"],
            "PowerDifference_kW": recomputed_power - fastfarm_rotor_power,
            "PowerError_pct": 100.0 * (recomputed_power / fastfarm_rotor_power - 1.0),
            "OptimizerMinusRecomputedPower_kW": command["power_kw"] - recomputed_power,
        })
    farm_power = pd.DataFrame(farm_rows)
    metrics = create_metrics(predictions, farm_power)

    direct_columns = join_keys + [
        "Mean_RotorEffectiveSpeed_mps", "TI_RotorEffective_pct",
        "FASTFarmInput_local_ws_mps", "FASTFarmInput_local_ti_pct", "Direct_TI_WasClipped",
        *[actual for actual, _ in DEL_CHANNELS.values()],
        *[f"{model}_Direct" for _, model in DEL_CHANNELS.values()],
    ]
    direct_output = predictions[direct_columns].copy()
    direct_output["DomainPolicy"] = "SATURATE_FASTFARM_TI_TO_4_14_PERCENT"

    predictions.to_csv(output_dir / "predictions_ALL_5T.csv", index=False)
    farm_power.to_csv(output_dir / "farm_power_ALL_5T.csv", index=False)
    metrics.to_csv(output_dir / "metrics_grouped_ALL_5T.csv", index=False)
    direct_output.to_csv(output_dir / "loads_direct_FASTFarm_inputs_ALL_5T.csv", index=False)

    summary = {
        **audit,
        "branches": {
            "CONNECTED": "ambient -> PyWake -> October base+sea DEL model",
            "DIRECT": "FAST.Farm local WS/TI -> October base+sea DEL model",
        },
        "output_files": [
            "predictions_ALL_5T.csv", "farm_power_ALL_5T.csv",
            "metrics_grouped_ALL_5T.csv", "loads_direct_FASTFarm_inputs_ALL_5T.csv",
            "summary_ALL_5T.json",
        ],
        "fairlead_definition": "maximum of FAST.Farm fairlead 1/2/3 DEL for each turbine row",
        "power_note": (
            "PyWake farm power is freshly recomputed at the actual FAST.Farm validation "
            "WS/TI. Metrics use FAST.Farm rotor power. The optimizer-stored centroid power "
            "is retained only as an audit column."
        ),
        "direct_domain_policy": (
            "For the direct diagnostic, measured FAST.Farm local TI is saturated to [4, 14] percent "
            "before load-model inference and its units are explicitly declared as percent. "
            "Original and model-input TI plus a clipping flag are retained. "
            "No out-of-domain extrapolation is enabled; the connected PyWake branch is unchanged."
        ),
    }
    (output_dir / "summary_ALL_5T.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nCompleted. Results: {output_dir}")
    for name in summary["output_files"]:
        print(f"  {output_dir / name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
