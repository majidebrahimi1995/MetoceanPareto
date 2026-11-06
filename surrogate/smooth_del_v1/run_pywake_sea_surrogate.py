#!/usr/bin/env python3
"""Run one PyWake farm case and the frozen DEL + sea-state surrogate.

The script produces, for every turbine:

1. PyWake local wind speed and turbulence intensity;
2. Mild and Normal isolated-surrogate predictions;
3. the two-anchor geometric interpolation at the requested Hs;
4. the frozen seven-sea-state Hs/Tp-corrected prediction.

It never reads FAST.Farm validation loads and never fits or updates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd


DEFAULT_MAIN = Path(__file__).resolve().parents[2]
EXPECTED_BASE_SHA256 = (
    "5260b2f50857f53f60a83faae56a210a251e3f51d9b684ab2456421c341ae679"
)
EXPECTED_SEA_SHA256 = (
    "2f57f175b33358c58fa03eb480b3b4165a66a5c2d8ba174971c00ef8f55c9676"
)

TURBINE_ALIASES = (
    "Turbine", "turbine", "turbine_id", "TurbineNo", "wt", "WT",
)
WS_ALIASES = (
    "local_ws_mps", "WS_eff_mps", "ws_eff_mps", "WS_eff",
    "effective_ws_mps", "EffectiveWS_mps", "wind_speed",
)
TI_ALIASES = (
    "local_ti_pct", "TI_eff_pct", "ti_eff_pct", "TI_eff",
    "effective_ti_pct", "EffectiveTI_pct", "turbulence_intensity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def comma_floats(text: str, name: str) -> np.ndarray:
    try:
        values = np.asarray([float(x.strip()) for x in text.split(",")], float)
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numbers") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    exact = {str(column).strip(): column for column in frame.columns}
    folded = {str(column).strip().casefold(): column for column in frame.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def normalize_inflow(
    frame: pd.DataFrame,
    n_turbines: int,
    ti_units: str,
) -> tuple[pd.DataFrame, str]:
    """Normalize a PyWake table to Turbine/local_ws_mps/local_ti_pct."""
    ws_col = find_column(frame, WS_ALIASES)
    ti_col = find_column(frame, TI_ALIASES)
    turbine_col = find_column(frame, TURBINE_ALIASES)
    if ws_col is None or ti_col is None:
        raise ValueError("table has no recognized PyWake WS/TI columns")

    result = pd.DataFrame({
        "local_ws_mps": pd.to_numeric(frame[ws_col], errors="coerce"),
        "local_ti_pct": pd.to_numeric(frame[ti_col], errors="coerce"),
    }).dropna()
    if turbine_col is not None:
        result.insert(
            0,
            "Turbine",
            pd.to_numeric(frame.loc[result.index, turbine_col], errors="coerce"),
        )
        result = result.dropna().sort_values("Turbine")
        result["Turbine"] = result["Turbine"].astype(int)
    else:
        result = result.reset_index(drop=True)
        result.insert(0, "Turbine", np.arange(1, len(result) + 1))

    result = result.reset_index(drop=True)
    if len(result) != n_turbines:
        raise ValueError(
            f"expected {n_turbines} turbine rows, found {len(result)}"
        )
    if result.Turbine.tolist() != list(range(1, n_turbines + 1)):
        raise ValueError("PyWake turbine identifiers must be 1..N exactly once")
    if not np.isfinite(result[["local_ws_mps", "local_ti_pct"]]).all().all():
        raise ValueError("PyWake WS/TI contains non-finite values")
    if (result[["local_ws_mps", "local_ti_pct"]] <= 0).any().any():
        raise ValueError("PyWake WS/TI must be positive")

    used_units = ti_units
    if ti_units == "fraction":
        result["local_ti_pct"] *= 100.0
        used_units = "fraction converted to percent"
    elif ti_units == "auto":
        if result.local_ti_pct.max() <= 1.0:
            result["local_ti_pct"] *= 100.0
            used_units = "fraction auto-converted to percent"
        else:
            used_units = "percent (auto-detected)"
    return result, used_units


def candidate_tables(text: str):
    lines = [line for line in text.splitlines() if line.strip()]
    for start, line in enumerate(lines):
        folded = line.casefold()
        has_ws = any(alias.casefold() in folded for alias in WS_ALIASES)
        has_ti = any(alias.casefold() in folded for alias in TI_ALIASES)
        if not (has_ws and has_ti):
            continue
        block = "\n".join(lines[start:])
        for kwargs in (
            {"sep": None, "engine": "python"},
            {"sep": r"\s+", "engine": "python"},
            {"sep": r"\s*\|\s*", "engine": "python"},
        ):
            try:
                yield pd.read_csv(io.StringIO(block), **kwargs)
            except Exception:
                pass


def parse_array(text: str, aliases: tuple[str, ...]) -> np.ndarray | None:
    alias_pattern = "|".join(re.escape(alias) for alias in aliases)
    match = re.search(
        rf"(?:{alias_pattern})\s*[:=]\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    values = re.split(r"[,\s]+", match.group(1).strip())
    try:
        return np.asarray([float(value) for value in values if value], float)
    except ValueError:
        return None


def parse_pywake_text(
    text: str,
    n_turbines: int,
    ti_units: str,
) -> tuple[pd.DataFrame, str]:
    for frame in candidate_tables(text):
        try:
            return normalize_inflow(frame, n_turbines, ti_units)
        except ValueError:
            pass

    ws = parse_array(text, WS_ALIASES)
    ti = parse_array(text, TI_ALIASES)
    if ws is not None and ti is not None and len(ws) == len(ti):
        frame = pd.DataFrame({
            "Turbine": np.arange(1, len(ws) + 1),
            "local_ws_mps": ws,
            "local_ti_pct": ti,
        })
        return normalize_inflow(frame, n_turbines, ti_units)

    raise ValueError(
        "Could not parse PyWake output. Supply --inflow-csv with columns "
        "Turbine,local_ws_mps,local_ti_pct; the raw PyWake output was saved."
    )


def model_domain_audit(
    base: dict,
    inflow: pd.DataFrame,
    yaw: np.ndarray,
    pitch: np.ndarray,
    hs: float,
    tp: float,
) -> dict[str, int]:
    bounds = base["bounds"]
    checks = {
        "ws": (inflow.local_ws_mps < bounds["ws"][0])
        | (inflow.local_ws_mps > bounds["ws"][1]),
        "ti": (inflow.local_ti_pct < bounds["ti"][0])
        | (inflow.local_ti_pct > bounds["ti"][1]),
        "yaw": (yaw < bounds["yaw"][0]) | (yaw > bounds["yaw"][1]),
        "pitch": (pitch < bounds["pitch"][0]) | (pitch > bounds["pitch"][1]),
        "sea": np.full(len(inflow), not (2.0 <= hs <= 6.0 and 8.0 <= tp <= 12.0)),
    }
    return {name: int(np.count_nonzero(flag)) for name, flag in checks.items()}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    bundle = DEFAULT_MAIN / "surrogate" / "smooth_del_v1"
    run = DEFAULT_MAIN / "models" / "october_20261013"
    p.add_argument("--farm", required=True)
    p.add_argument("--ws", type=float, required=True, help="ambient WS [m/s]")
    p.add_argument("--wd", type=float, required=True, help="physical WD [deg]")
    p.add_argument("--ti", type=float, required=True, help="ambient TI [percent]")
    p.add_argument("--hs", type=float, required=True, help="significant wave height [m]")
    p.add_argument("--tp", type=float, required=True, help="peak period [s]")
    p.add_argument("--yaw", required=True, help="comma-separated turbine yaw [deg]")
    p.add_argument("--pitch", required=True, help="comma-separated turbine pitch [deg]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, default=bundle)
    p.add_argument(
        "--pywake-script", type=Path,
        default=DEFAULT_MAIN / "pywake" / "bin" / "run_case.sh",
    )
    p.add_argument(
        "--base-model", type=Path,
        default=run / "base.joblib",
    )
    p.add_argument(
        "--sea-model", type=Path,
        default=run / "sea.joblib",
    )
    p.add_argument(
        "--inflow-csv", type=Path,
        help="reuse a PyWake table instead of running run_case.sh",
    )
    p.add_argument(
        "--pywake-ti-units", choices=("percent", "fraction", "auto"),
        default="percent",
    )
    p.add_argument(
        "--allow-outside-domain", action="store_true",
        help="predict outside the frozen training domain (not recommended)",
    )
    p.add_argument(
        "--allow-model-change", action="store_true",
        help="allow model hashes other than the frozen validation models",
    )
    return p



class ModelDomainError(ValueError):
    """A candidate violates the domain enforced by the canonical connected runner."""


class ConnectedEvaluator:
    """One canonical response evaluator for the CLI and optimizer.

    This class owns model loading and delegates all farm/layout/rotation/wake
    calculations to the existing run_frozen_pywake_case module.
    """

    def __init__(self, *, base_model, sea_model, bundle=None, pywake_runner=None,
                 allow_model_change=False):
        bundle = Path(bundle) if bundle is not None else Path(__file__).resolve().parent
        args = SimpleNamespace(base_model=Path(base_model), sea_model=Path(sea_model),
                               bundle=bundle, allow_model_change=allow_model_change)
        for path, description in (
            (args.base_model, "base model"),
            (args.sea_model, "sea model"),
            (args.bundle / "surrogate_core.py", "surrogate_core.py"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {description}: {path}")

        base_hash = sha256(args.base_model)
        sea_hash = sha256(args.sea_model)
        if not args.allow_model_change:
            if base_hash != EXPECTED_BASE_SHA256:
                raise ValueError("Base model hash is not the frozen October model")
            if sea_hash != EXPECTED_SEA_SHA256:
                raise ValueError("Sea model hash is not the frozen October model")


        sys.path.insert(0, str(args.bundle))
        from surrogate_core import TARGETS, predict_base, predict_continuous_sea

        if Path(sys.modules['surrogate_core'].__file__).resolve() != (args.bundle / 'surrogate_core.py').resolve():
            raise RuntimeError('surrogate_core was imported from a different bundle')

        base = joblib.load(args.base_model)
        sea = joblib.load(args.sea_model)
        if base.get("input_mode") != "avg9" or sea.get("input_mode") != "avg9":
            raise ValueError("Both frozen models must use the avg9 input definition")


        self.base, self.sea = base, sea
        self.targets = tuple(TARGETS)
        self._predict_base, self._predict_sea = predict_base, predict_continuous_sea
        self.base_hash, self.sea_hash = base_hash, sea_hash
        self.bundle = args.bundle.resolve()
        self.base_path, self.sea_path = args.base_model.resolve(), args.sea_model.resolve()
        self.pywake_runner = Path(pywake_runner or DEFAULT_MAIN / "pywake/bin/run_frozen_pywake_case.py").resolve()
        self._wake = None
        self._wake_module = None

    def _load_wake(self):
        if self._wake is None:
            if not self.pywake_runner.is_file():
                raise FileNotFoundError(self.pywake_runner)
            name = "_canonical_pywake_" + hashlib.sha256(str(self.pywake_runner).encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(name, self.pywake_runner)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if not hasattr(module, "FrozenPyWakeEvaluator"):
                raise RuntimeError("Install the matching canonical PyWake runner refactor first")
            self._wake_module = module
            self._wake = module.FrozenPyWakeEvaluator()
        return self._wake

    def pywake_frame(self, *, farm, ws, wd, ti, yaw, pitch):
        wake = self._load_wake()
        try:
            rows = wake.evaluate(farm=farm, ws=ws, wd=wd, ti=ti, yaw=yaw, pitch=pitch)
        except self._wake_module.PyWakeDomainError as exc:
            raise ModelDomainError(str(exc)) from exc
        except ValueError as exc:
            if "out of bounds" in str(exc).lower() or "outside" in str(exc).lower():
                raise ModelDomainError(str(exc)) from exc
            raise
        frame = pd.DataFrame(rows)
        required = ["WS_eff_mps", "TI_eff_pct", "Power_kW"]
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ModelDomainError("Non-finite canonical PyWake response")
        if (frame.Power_kW < 0).any():
            raise ModelDomainError("Negative canonical turbine power")
        return frame

    def predict_inflow(self, args, inflow, yaw, pitch):
        base, sea = self.base, self.sea
        TARGETS = self.targets
        predict_base, predict_continuous_sea = self._predict_base, self._predict_sea
        audit = model_domain_audit(base, inflow, yaw, pitch, args.hs, args.tp)
        if sum(audit.values()) and not args.allow_outside_domain:
            raise ModelDomainError(
                f"Inputs outside the frozen training domain: {audit}. "
                "No DEL prediction was written."
            )

        n = len(inflow)
        ws_local = inflow.local_ws_mps.to_numpy(float)
        ti_local = inflow.local_ti_pct.to_numpy(float)
        wd = np.full(n, args.wd)
        hs = np.full(n, args.hs)
        tp = np.full(n, args.tp)
        common = (ws_local, ti_local, wd, yaw, pitch)

        mild = np.asarray(predict_base(base, *common, "Mild"), float)
        normal = np.asarray(predict_base(base, *common, "Normal"), float)
        weight = ((hs - 2.0) / 4.0)[:, None]
        anchor = np.exp((1.0 - weight) * np.log(mild) + weight * np.log(normal))
        corrected = np.asarray(
            predict_continuous_sea(base, sea, *common, hs, tp), float
        )

        expected_shape = (n, len(TARGETS))
        for name, values in (
            ("mild", mild), ("normal", normal),
            ("anchor", anchor), ("corrected", corrected),
        ):
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f"Unexpected {name} prediction shape/value: {values.shape}")
        if (mild <= 0).any() or (normal <= 0).any() or (corrected <= 0).any():
            raise ValueError("DEL predictions must be strictly positive")

        result = pd.DataFrame({
            "Farm": args.farm,
            "Turbine": inflow.Turbine,
            "ambient_ws_mps": args.ws,
            "ambient_ti_pct": args.ti,
            "physical_wd_deg": args.wd,
            "local_ws_mps": ws_local,
            "local_ti_pct": ti_local,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "hs_m": args.hs,
            "tp_s": args.tp,
        })
        for j, target in enumerate(TARGETS):
            result[f"mild_{target}"] = mild[:, j]
            result[f"normal_{target}"] = normal[:, j]
            result[f"anchor_{target}"] = anchor[:, j]
            result[f"sea_{target}"] = corrected[:, j]
            result[f"sea_factor_{target}"] = corrected[:, j] / anchor[:, j]

        return result, audit

    def evaluate(self, *, farm, ws, wd, ti, hs, tp, yaw, pitch, allow_outside_domain=False):
        yaw, pitch = np.asarray(yaw, float).reshape(-1), np.asarray(pitch, float).reshape(-1)
        if len(yaw) != len(pitch) or not len(yaw) or not np.isfinite(np.r_[yaw, pitch]).all():
            raise ModelDomainError("Yaw/pitch vectors must have equal nonzero finite lengths")
        if not np.isfinite([ws, wd, ti, hs, tp]).all():
            raise ModelDomainError("Environmental inputs must be finite")
        aero = self.pywake_frame(farm=farm, ws=ws, wd=wd, ti=ti, yaw=yaw, pitch=pitch)
        inflow, _ = normalize_inflow(aero, len(yaw), "percent")
        args = SimpleNamespace(farm=farm, ws=ws, wd=wd, ti=ti, hs=hs, tp=tp,
                               allow_outside_domain=allow_outside_domain)
        result, audit = self.predict_inflow(args, inflow, yaw, pitch)
        return {"frame": result, "pywake": aero, "audit": audit}

    def evaluate_many(self, *, actions, **environment):
        actions = np.asarray(actions, float)
        if actions.ndim != 2 or actions.shape[1] % 2:
            raise ValueError("Actions must be (batch, 2*N), yaw first, then pitch")
        n = actions.shape[1] // 2
        return [self.evaluate(yaw=a[:n], pitch=a[n:], **environment) for a in actions]

    def provenance(self):
        wake = self._load_wake()
        stack = wake.stack
        paths = [self.base_path, self.sea_path, self.bundle / "surrogate_core.py",
                 Path(__file__).resolve(), self.pywake_runner]
        paths += list((stack / "runtime").rglob("*.py"))
        paths += [stack / "pywake" / name for name in (
            "pywake_absolute3d_calibration.json",
            "openfast_absolute_3d_pywake_table_TI6_PROJECTED_CT_BOUNDED.csv",
            "pitch_ct_tilt_map.csv")]
        return {"asset_files": [str(p.resolve()) for p in paths],
                "resolved_stack": str(stack),
                "model_sha256": {"base": self.base_hash, "sea": self.sea_hash}}

def main() -> None:
    args = parser().parse_args()
    yaw = comma_floats(args.yaw, "--yaw")
    pitch = comma_floats(args.pitch, "--pitch")
    if len(yaw) != len(pitch):
        raise ValueError("--yaw and --pitch must contain the same number of turbines")

    evaluator = ConnectedEvaluator(base_model=args.base_model, sea_model=args.sea_model,
                                   bundle=args.bundle, allow_model_change=args.allow_model_change)
    base_hash, sea_hash = evaluator.base_hash, evaluator.sea_hash
    TARGETS = evaluator.targets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".pywake.txt")
    pywake_csv_path = args.output.with_suffix(".pywake.csv")
    command = None
    if args.inflow_csv:
        inflow, ti_units = normalize_inflow(
            pd.read_csv(args.inflow_csv), len(yaw), args.pywake_ti_units
        )
    elif args.pywake_script.resolve() == (DEFAULT_MAIN / "pywake/bin/run_case.sh").resolve():
        aerodynamic = evaluator.pywake_frame(farm=args.farm, ws=args.ws, wd=args.wd, ti=args.ti,
                                            yaw=yaw, pitch=pitch)
        aerodynamic.to_csv(pywake_csv_path, index=False)
        raw_path.write_text("Canonical in-process PyWake evaluator\n" + aerodynamic.to_csv(index=False), encoding="utf-8")
        inflow, ti_units = normalize_inflow(aerodynamic, len(yaw), "percent")
    else:
        if not args.pywake_script.is_file():
            raise FileNotFoundError(f"Missing PyWake runner: {args.pywake_script}")
        command = [
            str(args.pywake_script),
            "--farm", args.farm,
            "--ws", str(args.ws),
            "--wd", str(args.wd),
            "--ti", str(args.ti),
            f"--yaw={args.yaw}",
            f"--pitch={args.pitch}",
            "--output", str(pywake_csv_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        raw_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(
                f"PyWake failed with exit code {completed.returncode}; see {raw_path}"
            )
        if pywake_csv_path.is_file():
            inflow, ti_units = normalize_inflow(
                pd.read_csv(pywake_csv_path), len(yaw), args.pywake_ti_units
            )
        else:
            inflow, ti_units = parse_pywake_text(
                completed.stdout, len(yaw), args.pywake_ti_units
            )

    result, audit = evaluator.predict_inflow(args, inflow, yaw, pitch)
    n = len(inflow)
    result.to_csv(args.output, index=False)

    metadata = {
        "protocol": "PyWake local inflow -> frozen avg9 base -> frozen Hs/Tp residual",
        "pywake_command": command,
        "pywake_raw_output": None if args.inflow_csv else str(raw_path),
        "pywake_csv": None if args.inflow_csv else str(pywake_csv_path),
        "reused_inflow_csv": None if not args.inflow_csv else str(args.inflow_csv),
        "pywake_ti_units": ti_units,
        "model_sha256": {"base": base_hash, "sea": sea_hash},
        "outside_training_domain_by_input": audit,
        "n_turbines": n,
        "targets": list(TARGETS),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    display = ["Turbine", "local_ws_mps", "local_ti_pct"]
    display.extend(f"sea_{target}" for target in TARGETS)
    print(result[display].to_string(index=False))
    print(f"\nSaved predictions: {args.output}")
    print(f"Saved provenance:  {metadata_path}")


if __name__ == "__main__":
    main()

