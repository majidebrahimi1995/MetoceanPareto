#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate PyWake against six-seed-averaged two-turbine FAST.Farm data.

Calibration boundary condition
------------------------------
For every aerodynamic/controller case, the measured FAST.Farm T1 values are
used as the PyWake free-stream boundary condition:

    WS_site = T1_Mean_RotorEffectiveSpeed_mps
    TI_site = T1_TI_RotorEffective_pct / 100

PyWake is then calibrated against:

    T2_Mean_RotorEffectiveSpeed_mps
    T2_TI_RotorEffective_pct

This isolates the T1 -> T2 wake transformation from differences between the
nominal metadata inflow and the wind actually experienced by T1.

The script:
  1. reads the FAST.Farm calibration CSV and OpenFAST pitch/Ct/Cp table;
  2. optionally averages identical aerodynamic cases across sea states;
  3. splits complete physical conditions into train/validation groups;
  4. compares Bastankhah and Zong Gaussian wake models;
  5. checks both PyWake yaw-sign conventions;
  6. calibrates wake recovery/width and yaw deflection against T2 WS;
  7. calibrates Crespo-Hernandez magnitude/decay against T2 TI;
  8. writes predictions, metrics, plots and a JSON configuration suitable for
     loading in the PyWake + load-surrogate environment.

The final load surrogate is NOT called here. This script only calibrates the
PyWake quantities that will later be passed to that surrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid nested BLAS/OpenMP parallelism when scipy uses multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# PyWake imports with compatibility fallbacks
# -----------------------------------------------------------------------------

try:
    import py_wake
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtNDTabular
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014
    try:
        from py_wake.literature.gaussian_models import Zong_PorteAgel_2020
    except ImportError:
        Zong_PorteAgel_2020 = None
    try:
        from py_wake.turbulence_models import CrespoHernandez
    except ImportError:
        from py_wake.turbulence_models.crespo import CrespoHernandez
    try:
        from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    except ImportError:
        from py_wake.deflection_models import JimenezWakeDeflection
    try:
        from py_wake.superposition_models import LinearSum
    except ImportError:
        LinearSum = None
except Exception as exc:  # pragma: no cover - cluster runtime check
    raise RuntimeError(
        "PyWake could not be imported. Run this script with the project's "
        "PyWake Python environment. Original error: " + repr(exc)
    ) from exc


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=json_default)
    tmp.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing_file(patterns: Sequence[str]) -> Optional[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(Path().glob(pattern))
    matches = sorted({p.resolve() for p in matches if p.is_file()})
    return matches[0] if matches else None


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def resolve_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    *,
    required: bool,
    label: str,
) -> Optional[str]:
    direct = {str(c): str(c) for c in df.columns}
    normalized = {normalize_name(c): str(c) for c in df.columns}
    for alias in aliases:
        if alias in direct:
            return direct[alias]
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(
            f"Missing required column for {label}. Tried {list(aliases)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def finite_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def expand_numeric(value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if len(arr) == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if len(arr) != n:
        raise ValueError(f"Expected length 1 or {n}, found {len(arr)}")
    return arr.astype(float)


# -----------------------------------------------------------------------------
# OpenFAST pitch/Ct/Cp lookup table
# -----------------------------------------------------------------------------

PITCH_TABLE_ALIASES = {
    "wind_speed": ["wind_speed", "windspeed", "ws", "WindSpeed", "WindSpeed_[m/s]"],
    "pitch_offset_deg": [
        "pitch_offset_deg", "pitch_deg", "blade_pitch_offset_deg", "beta_deg",
        "Beta", "pitch", "Pitch", "BldPitch", "collective_pitch_deg",
    ],
    "ct": ["ct", "Ct", "CT"],
    "cp": ["cp", "Cp", "CP"],
    "power_w": ["power_w", "Power_W", "power", "Power", "P_w"],
    "power_kw": ["power_kw", "Power_kW", "P_kW"],
    "power_mw": ["power_mw", "Power_MW", "P_MW"],
    "platform_pitch_deg": [
        "platform_pitch_deg", "tilt_deg", "tilt", "Tilt", "platform_tilt_deg",
        "platform_pitch", "PtfmPitch", "PtfmPitch_deg",
    ],
}


def find_pitch_col(df: pd.DataFrame, canonical: str, required: bool) -> Optional[str]:
    return resolve_column(
        df,
        PITCH_TABLE_ALIASES[canonical],
        required=required,
        label=f"pitch-table {canonical}",
    )


def read_pitch_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Pitch/Ct table not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    ws_col = find_pitch_col(raw, "wind_speed", True)
    pitch_col = find_pitch_col(raw, "pitch_offset_deg", True)
    ct_col = find_pitch_col(raw, "ct", True)
    cp_col = find_pitch_col(raw, "cp", False)
    pw_col = find_pitch_col(raw, "power_w", False)
    pkw_col = find_pitch_col(raw, "power_kw", False)
    pmw_col = find_pitch_col(raw, "power_mw", False)
    platform_col = find_pitch_col(raw, "platform_pitch_deg", False)

    out = pd.DataFrame({
        "wind_speed": pd.to_numeric(raw[ws_col], errors="coerce"),
        "pitch_offset_deg": pd.to_numeric(raw[pitch_col], errors="coerce"),
        "ct": pd.to_numeric(raw[ct_col], errors="coerce"),
    })
    if cp_col is not None:
        out["cp"] = pd.to_numeric(raw[cp_col], errors="coerce")
    if pw_col is not None:
        out["power_w"] = pd.to_numeric(raw[pw_col], errors="coerce")
    elif pkw_col is not None:
        out["power_w"] = 1e3 * pd.to_numeric(raw[pkw_col], errors="coerce")
    elif pmw_col is not None:
        out["power_w"] = 1e6 * pd.to_numeric(raw[pmw_col], errors="coerce")
    if platform_col is not None:
        out["platform_pitch_deg"] = pd.to_numeric(raw[platform_col], errors="coerce")

    if "cp" not in out and "power_w" not in out:
        raise ValueError("Pitch table must contain Cp or power.")

    required = ["wind_speed", "pitch_offset_deg", "ct"]
    required.append("power_w" if "power_w" in out else "cp")
    out = out.dropna(subset=required)
    out = (
        out.groupby(["wind_speed", "pitch_offset_deg"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["wind_speed", "pitch_offset_deg"])
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("No valid rows remained in the pitch table.")
    return out


def pivot_complete(
    df: pd.DataFrame,
    value_col: str,
    ws_grid: np.ndarray,
    pitch_grid: np.ndarray,
) -> np.ndarray:
    pivot = (
        df.pivot_table(index="wind_speed", columns="pitch_offset_deg", values=value_col)
        .reindex(index=ws_grid, columns=pitch_grid)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    pivot = pivot.interpolate(axis=0, limit_direction="both")
    pivot = pivot.interpolate(axis=1, limit_direction="both")
    if pivot.isna().any().any():
        raise ValueError(f"Pitch table column {value_col!r} is not a complete grid.")
    return pivot.to_numpy(dtype=float)


@dataclass
class PitchTableModel:
    turbine: Any
    table: pd.DataFrame
    ws_grid: np.ndarray
    pitch_grid: np.ndarray
    platform_pitch_table: Optional[np.ndarray]

    def platform_pitch(self, ws: Sequence[float], pitch: Sequence[float]) -> np.ndarray:
        ws_arr = np.asarray(ws, dtype=float).reshape(-1)
        p_arr = np.asarray(pitch, dtype=float).reshape(-1)
        if self.platform_pitch_table is None:
            return np.zeros_like(ws_arr)
        ws_arr = np.clip(ws_arr, self.ws_grid[0], self.ws_grid[-1])
        p_arr = np.clip(p_arr, self.pitch_grid[0], self.pitch_grid[-1])
        result = np.empty_like(ws_arr)
        for k, (w, p) in enumerate(zip(ws_arr, p_arr)):
            iw = max(0, min(len(self.ws_grid) - 2, np.searchsorted(self.ws_grid, w) - 1))
            ip = max(0, min(len(self.pitch_grid) - 2, np.searchsorted(self.pitch_grid, p) - 1))
            if len(self.ws_grid) == 1:
                iw = 0
            if len(self.pitch_grid) == 1:
                ip = 0
            iw1 = min(iw + 1, len(self.ws_grid) - 1)
            ip1 = min(ip + 1, len(self.pitch_grid) - 1)
            w0, w1 = self.ws_grid[iw], self.ws_grid[iw1]
            p0, p1 = self.pitch_grid[ip], self.pitch_grid[ip1]
            fw = 0.0 if abs(w1 - w0) < 1e-12 else (w - w0) / (w1 - w0)
            fp = 0.0 if abs(p1 - p0) < 1e-12 else (p - p0) / (p1 - p0)
            q00 = self.platform_pitch_table[iw, ip]
            q10 = self.platform_pitch_table[iw1, ip]
            q01 = self.platform_pitch_table[iw, ip1]
            q11 = self.platform_pitch_table[iw1, ip1]
            result[k] = (
                (1 - fw) * (1 - fp) * q00
                + fw * (1 - fp) * q10
                + (1 - fw) * fp * q01
                + fw * fp * q11
            )
        return result


@lru_cache(maxsize=4)
def build_pitch_table_model_cached(
    table_path_text: str,
    diameter: float,
    hub_height: float,
    rho: float,
) -> PitchTableModel:
    path = Path(table_path_text)
    df = read_pitch_table(path)
    ws_grid = np.sort(df["wind_speed"].unique().astype(float))
    pitch_grid = np.sort(df["pitch_offset_deg"].unique().astype(float))
    ct_table = pivot_complete(df, "ct", ws_grid, pitch_grid)
    if "power_w" in df:
        power_w = pivot_complete(df, "power_w", ws_grid, pitch_grid)
    else:
        cp = pivot_complete(df, "cp", ws_grid, pitch_grid)
        area = math.pi * (diameter / 2.0) ** 2
        power_w = 0.5 * rho * area * cp * ws_grid[:, None] ** 3
    platform_table = None
    if "platform_pitch_deg" in df:
        platform_table = pivot_complete(df, "platform_pitch_deg", ws_grid, pitch_grid)

    pct = PowerCtNDTabular(
        input_keys=["ws", "pitch_offset_deg"],
        value_lst=[ws_grid, pitch_grid],
        power_arr=power_w,
        power_unit="w",
        ct_arr=ct_table,
    )
    turbine = WindTurbine(
        name="IEA15MW_OpenFAST_pitch_table",
        diameter=float(diameter),
        hub_height=float(hub_height),
        powerCtFunction=pct,
    )
    return PitchTableModel(turbine, df, ws_grid, pitch_grid, platform_table)


# -----------------------------------------------------------------------------
# Wake components
# -----------------------------------------------------------------------------


class ScaledJimenezWakeDeflection(JimenezWakeDeflection):
    """Scale Jimenez horizontal/vertical wake displacement at runtime."""

    def __init__(self, *args: Any, deflection_scale: float = 1.0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.deflection_scale = float(deflection_scale)

    def calc_deflection(
        self,
        dw_ijlk: np.ndarray,
        hcw_ijlk: np.ndarray,
        dh_ijlk: np.ndarray,
        yaw_ilk: np.ndarray,
        tilt_ilk: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        out = super().calc_deflection(
            dw_ijlk=dw_ijlk,
            hcw_ijlk=hcw_ijlk,
            dh_ijlk=dh_ijlk,
            yaw_ilk=yaw_ilk,
            tilt_ilk=tilt_ilk,
            **kwargs,
        )
        if np.isclose(self.deflection_scale, 1.0):
            return out
        if isinstance(out, tuple) and len(out) == 3:
            dw_out, hcw_out, dh_out = out
            s = self.deflection_scale
            return (
                dw_out,
                hcw_ijlk + s * (hcw_out - hcw_ijlk),
                dh_ijlk + s * (dh_out - dh_ijlk),
            )
        raise RuntimeError(f"Unexpected Jimenez output: {type(out)}")


def import_symbol(module_names: Sequence[str], symbol: str) -> Any:
    errors = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(f"Could not import {symbol}. Attempts: {errors}")


def make_rotor_average(name: str) -> Any:
    name = name.lower().strip()
    if name in {"center", "none", "rotorcenter"}:
        return None

    GQGridRotorAvg = import_symbol(
        [
            "py_wake.rotor_avg_models.rotor_avg_model",
            "py_wake.rotor_avg_models",
        ],
        "GQGridRotorAvg",
    )
    if name == "gq":
        return GQGridRotorAvg(4, 3)

    if name == "gaussian_overlap":
        GaussianOverlapAvgModel = import_symbol(
            [
                "py_wake.rotor_avg_models.gaussian_overlap_model",
                "py_wake.rotor_avg_models",
            ],
            "GaussianOverlapAvgModel",
        )
        return GaussianOverlapAvgModel()

    if name == "gq_power3":
        WSPowerRotorAvg = import_symbol(
            [
                "py_wake.rotor_avg_models.rotor_avg_model",
                "py_wake.rotor_avg_models",
            ],
            "WSPowerRotorAvg",
        )
        base = GQGridRotorAvg(4, 3)
        attempts = [
            lambda: WSPowerRotorAvg(rotorAvgModel=base, alpha=3),
            lambda: WSPowerRotorAvg(base, alpha=3),
            lambda: WSPowerRotorAvg(base, 3),
        ]
        last_error: Optional[Exception] = None
        for attempt in attempts:
            try:
                return attempt()
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not construct WSPowerRotorAvg: {last_error}")

    raise ValueError(f"Unknown rotor-average model: {name}")


def make_crespo(c0: float, c3: float) -> Any:
    c = [float(c0), 0.8325, 0.0325, float(c3)]
    try:
        return CrespoHernandez(c=c)
    except TypeError:
        # Some older versions use positional c.
        return CrespoHernandez(c)


# -----------------------------------------------------------------------------
# Calibration data
# -----------------------------------------------------------------------------

CALIBRATION_ALIASES: Dict[str, Sequence[str]] = {
    "x1": ["T1_X", "x1", "WT1_X"],
    "y1": ["T1_Y", "y1", "WT1_Y"],
    "x2": ["T2_X", "x2", "WT2_X"],
    "y2": ["T2_Y", "y2", "WT2_Y"],
    "yaw1": ["Yaw_T1", "Controller_Yaw_T1", "yaw1", "T1_Yaw"],
    "yaw2": ["Yaw_T2", "Controller_Yaw_T2", "yaw2", "T2_Yaw"],
    "pitch1": ["Pitch_T1", "Controller_Pitch_T1", "pitch1", "T1_Pitch"],
    "pitch2": ["Pitch_T2", "Controller_Pitch_T2", "pitch2", "T2_Pitch"],
    "t1_ws": [
        "T1_Mean_RotorEffectiveSpeed_mps",
        "T1_Mean_RotorUx_mps",
        "T1_effective_ws",
    ],
    "t2_ws": [
        "T2_Mean_RotorEffectiveSpeed_mps",
        "T2_Mean_RotorUx_mps",
        "T2_effective_ws",
    ],
    "t1_ti_pct": ["T1_TI_RotorEffective_pct", "T1_effective_ti_pct"],
    "t2_ti_pct": ["T2_TI_RotorEffective_pct", "T2_effective_ti_pct"],
    "ambient_ws": ["WS", "AmbientWS_mps", "wind_speed", "windspeed"],
    "ambient_ti_pct": ["TI", "AmbientTI_pct", "TI_percent"],
    "wd_physical": ["WD_physical_deg", "WD_deg", "WindDirection_deg"],
    "rotation": ["Rotation_deg", "LayoutRotation_deg"],
    "controller_row": ["ControllerRow", "ControllerRowReference", "SegmentPosition"],
    "group_id": ["GroupID", "PhysicalGroupID"],
    "sea_state": ["SeaState", "SeaCondition", "seaCondition", "Sea_State"],
    "hs": ["Hs", "Hs_m", "WaveHs", "SignificantWaveHeight_m"],
    "tp": ["Tp", "Tp_s", "WaveTp", "PeakPeriod_s"],
}

REQUIRED_CALIBRATION = [
    "x1", "y1", "x2", "y2", "yaw1", "pitch1",
    "t1_ws", "t2_ws", "t1_ti_pct", "t2_ti_pct",
]

OPTIONAL_DEFAULT_ZERO = ["yaw2", "pitch2"]
OPTIONAL_METADATA = [
    "ambient_ws", "ambient_ti_pct", "wd_physical", "rotation",
    "controller_row", "group_id", "sea_state", "hs", "tp",
]


def load_and_prepare_data(
    path: Path,
    *,
    average_sea_state: bool,
    wd_mode: str,
    fixed_wd: float,
    max_rows: Optional[int],
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration data file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, sheet_name="Sheet1")
    elif suffix == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported calibration-data format: {path.suffix}. "
            "Supported formats are .csv, .xlsx, and .xls."
        )
    
    mapping: Dict[str, Optional[str]] = {}

    for key in REQUIRED_CALIBRATION:
        mapping[key] = resolve_column(
            raw, CALIBRATION_ALIASES[key], required=True, label=key
        )
    for key in OPTIONAL_DEFAULT_ZERO + OPTIONAL_METADATA:
        mapping[key] = resolve_column(
            raw, CALIBRATION_ALIASES[key], required=False, label=key
        )

    data = pd.DataFrame(index=raw.index)
    for key in REQUIRED_CALIBRATION + OPTIONAL_DEFAULT_ZERO:
        source = mapping[key]
        if source is None:
            data[key] = 0.0
        else:
            data[key] = pd.to_numeric(raw[source], errors="coerce")

    for key in OPTIONAL_METADATA:
        source = mapping[key]
        if source is None:
            continue
        if key == "sea_state":
            data[key] = raw[source].astype(str)
        else:
            data[key] = pd.to_numeric(raw[source], errors="coerce")

    data["source_row"] = np.arange(len(data), dtype=int)

    numeric_required = REQUIRED_CALIBRATION + OPTIONAL_DEFAULT_ZERO
    valid = np.ones(len(data), dtype=bool)
    for key in numeric_required:
        valid &= np.isfinite(data[key].to_numpy(dtype=float))
    valid &= data["t1_ws"].to_numpy(dtype=float) > 0
    valid &= data["t2_ws"].to_numpy(dtype=float) > 0
    valid &= data["t1_ti_pct"].to_numpy(dtype=float) >= 0
    valid &= data["t2_ti_pct"].to_numpy(dtype=float) >= 0
    dropped = int((~valid).sum())
    data = data.loc[valid].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid calibration rows remained after numeric cleaning.")

    # Coordinates are normally already rotated in the metadata, hence fixed 270 deg.
    if wd_mode == "fixed":
        data["pywake_wd"] = float(fixed_wd)
    elif wd_mode == "metadata":
        if "wd_physical" not in data:
            raise KeyError("--wd-mode metadata requires a wind-direction column.")
        data["pywake_wd"] = np.mod(data["wd_physical"].to_numpy(dtype=float), 360.0)
    else:
        raise ValueError(wd_mode)

    # Average only across sea-state repetitions while preserving all aerodynamic
    # and controller inputs. This is deliberately explicit, not a broad group-by
    # over every non-target column.
    aero_keys = ["x1", "y1", "x2", "y2", "yaw1", "yaw2", "pitch1", "pitch2", "pywake_wd"]
    for key in ["ambient_ws", "ambient_ti_pct", "wd_physical", "rotation", "controller_row"]:
        if key in data:
            aero_keys.append(key)

    if average_sea_state and any(k in data for k in ["sea_state", "hs", "tp"]):
        value_cols = ["t1_ws", "t2_ws", "t1_ti_pct", "t2_ti_pct"]
        grouped = data.groupby(aero_keys, dropna=False, as_index=False)
        means = grouped[value_cols].mean()
        counts = grouped.size().rename(columns={"size": "sea_rows_averaged"})
        data = means.merge(counts, on=aero_keys, how="left")
    else:
        data["sea_rows_averaged"] = 1

    # Physical split group: controller states must remain together. Exclude yaw,
    # pitch and controller-row index from this key.
    split_keys = ["x1", "y1", "x2", "y2", "pywake_wd"]
    for key in ["ambient_ws", "ambient_ti_pct", "wd_physical", "rotation"]:
        if key in data:
            split_keys.append(key)
    rounded = data[split_keys].copy()
    for col in split_keys:
        rounded[col] = pd.to_numeric(rounded[col], errors="coerce").round(8)
    data["physical_group"] = pd.util.hash_pandas_object(rounded, index=False).astype(str)

    if max_rows is not None and max_rows > 0 and len(data) > max_rows:
        rng = np.random.default_rng(random_seed)
        groups = data["physical_group"].unique()
        rng.shuffle(groups)
        chosen: List[str] = []
        count = 0
        for group in groups:
            chosen.append(group)
            count += int((data["physical_group"] == group).sum())
            if count >= max_rows:
                break
        data = data[data["physical_group"].isin(chosen)].reset_index(drop=True)

    dx = data["x2"].to_numpy() - data["x1"].to_numpy()
    dy = data["y2"].to_numpy() - data["y1"].to_numpy()
    wd_rad = np.deg2rad(data["pywake_wd"].to_numpy())
    flow_x = -np.sin(wd_rad)
    flow_y = -np.cos(wd_rad)
    data["downwind_m"] = dx * flow_x + dy * flow_y
    data["crosswind_m"] = -dx * flow_y + dy * flow_x
    data["spacing_D"] = np.hypot(dx, dy) / 240.0
    data["downwind_D"] = data["downwind_m"] / 240.0

    if (data["downwind_m"] <= 0).any():
        bad = int((data["downwind_m"] <= 0).sum())
        raise ValueError(
            f"{bad} rows place T2 upstream of T1 under the selected PyWake WD. "
            "For rotated coordinates use --wd-mode fixed --fixed-wd 270. "
            "For unrotated coordinates use --wd-mode metadata."
        )

    print(f"Loaded calibration rows       : {len(raw)}")
    print(f"Dropped invalid rows          : {dropped}")
    print(f"Rows after sea-state handling : {len(data)}")
    print(f"Physical split groups         : {data['physical_group'].nunique()}")
    print(
        "Spacing range              : "
        f"{data['spacing_D'].min():.3f}D to {data['spacing_D'].max():.3f}D"
    )
    return data, mapping, split_keys


def split_train_validation(
    data: pd.DataFrame,
    validation_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    groups = data["physical_group"].unique().copy()
    if len(groups) < 2:
        raise ValueError("At least two physical groups are needed for validation.")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(validation_fraction * len(groups))))
    n_val = min(n_val, len(groups) - 1)
    val_groups = set(groups[:n_val])
    out = data.copy()
    out["split"] = np.where(out["physical_group"].isin(val_groups), "validation", "train")
    print(f"Training rows/groups          : {(out.split == 'train').sum()} / {out.loc[out.split == 'train', 'physical_group'].nunique()}")
    print(f"Validation rows/groups        : {(out.split == 'validation').sum()} / {out.loc[out.split == 'validation', 'physical_group'].nunique()}")
    return out


def dataframe_to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    keys = [
        "x1", "y1", "x2", "y2", "yaw1", "yaw2", "pitch1", "pitch2",
        "t1_ws", "t2_ws", "t1_ti_pct", "t2_ti_pct", "pywake_wd",
        "spacing_D", "downwind_D", "crosswind_m",
    ]
    return {key: df[key].to_numpy(dtype=float) for key in keys}


# -----------------------------------------------------------------------------
# PyWake prediction
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    model: str
    rotor_model: str
    yaw_sign: int
    tilt_mode: str
    fixed_tilt_deg: float
    tilt_iterations: int
    diameter: float
    hub_height: float
    rho: float
    pitch_table: str

    @property
    def name(self) -> str:
        return f"{self.model}__{self.rotor_model}__yaw{self.yaw_sign:+d}"


def compute_tilt(
    pitch_model: PitchTableModel,
    ws_guess: np.ndarray,
    pitch: np.ndarray,
    mode: str,
    fixed_tilt_deg: float,
) -> np.ndarray:
    mode = mode.lower()
    platform_pitch = pitch_model.platform_pitch(ws_guess, pitch)
    if mode == "zero":
        return np.zeros_like(pitch)
    if mode == "fixed6" or mode == "fixed":
        return np.full_like(pitch, fixed_tilt_deg)
    if mode in {"csv", "csv_platform", "csv_only"}:
        return platform_pitch
    if mode == "six_plus_csv":
        return fixed_tilt_deg + platform_pitch
    if mode == "six_minus_csv":
        return fixed_tilt_deg - platform_pitch
    raise ValueError(f"Unknown tilt mode: {mode}")


def build_wfm(
    spec: CandidateSpec,
    site: Any,
    wake_params: Sequence[float],
    crespo_c0: float,
    crespo_c3: float,
) -> Any:
    rotor_avg = make_rotor_average(spec.rotor_model)
    deflection = ScaledJimenezWakeDeflection(deflection_scale=float(wake_params[2]))
    turbulence = make_crespo(crespo_c0, crespo_c3)
    common = dict(
        rotorAvgModel=rotor_avg,
        deflectionModel=deflection,
        turbulenceModel=turbulence,
    )
    if LinearSum is not None:
        common["superpositionModel"] = LinearSum()

    model = spec.model.lower()
    if model == "bastankhah":
        k, ceps, _ = map(float, wake_params)
        return Bastankhah_PorteAgel_2014(
            site,
            build_pitch_table_model_cached(
                spec.pitch_table, spec.diameter, spec.hub_height, spec.rho
            ).turbine,
            k=k,
            ceps=ceps,
            **common,
        )
    if model == "zong":
        if Zong_PorteAgel_2020 is None:
            raise ImportError("Zong_PorteAgel_2020 is unavailable in this PyWake version.")
        k_eff, eps_coeff, _ = map(float, wake_params)
        return Zong_PorteAgel_2020(
            site,
            build_pitch_table_model_cached(
                spec.pitch_table, spec.diameter, spec.hub_height, spec.rho
            ).turbine,
            a=[0.0, k_eff],
            eps_coeff=eps_coeff,
            **common,
        )
    raise ValueError(f"Unknown model: {spec.model}")


def predict_one(
    arrays: Mapping[str, np.ndarray],
    index: int,
    spec: CandidateSpec,
    wake_params: Sequence[float],
    crespo_c0: float,
    crespo_c3: float,
) -> Tuple[float, float]:
    pitch_model = build_pitch_table_model_cached(
        spec.pitch_table, spec.diameter, spec.hub_height, spec.rho
    )
    ambient_ws = float(arrays["t1_ws"][index])
    ambient_ti = float(arrays["t1_ti_pct"][index]) / 100.0
    site = UniformSite(p_wd=[1.0], ti=ambient_ti)
    wfm = build_wfm(spec, site, wake_params, crespo_c0, crespo_c3)

    x = np.array([arrays["x1"][index], arrays["x2"][index]], dtype=float)
    y = np.array([arrays["y1"][index], arrays["y2"][index]], dtype=float)
    x -= x[0]
    y -= y[0]
    yaw = spec.yaw_sign * np.array(
        [arrays["yaw1"][index], arrays["yaw2"][index]], dtype=float
    )
    pitch = np.array(
        [arrays["pitch1"][index], arrays["pitch2"][index]], dtype=float
    )
    yaw_ilk = yaw[:, None, None]
    pitch_ilk = pitch[:, None, None]

    ws_guess = np.full(2, ambient_ws, dtype=float)
    sim = None
    n_iter = max(1, int(spec.tilt_iterations))
    if spec.tilt_mode.lower() in {"zero", "fixed", "fixed6"}:
        n_iter = 1
    for _ in range(n_iter):
        tilt = compute_tilt(
            pitch_model,
            ws_guess,
            pitch,
            spec.tilt_mode,
            spec.fixed_tilt_deg,
        )
        sim = wfm(
            x,
            y,
            wd=[float(arrays["pywake_wd"][index])],
            ws=[ambient_ws],
            yaw=yaw_ilk,
            tilt=tilt[:, None, None],
            pitch_offset_deg=pitch_ilk,
        )
        ws_values = np.asarray(sim.WS_eff.values, dtype=float).squeeze().reshape(-1)
        if len(ws_values) != 2:
            raise RuntimeError(f"Unexpected WS_eff shape: {sim.WS_eff.shape}")
        ws_guess = np.clip(ws_values, pitch_model.ws_grid[0], pitch_model.ws_grid[-1])

    assert sim is not None
    ws_values = np.asarray(sim.WS_eff.values, dtype=float).squeeze().reshape(-1)
    ti_values = 100.0 * np.asarray(sim.TI_eff.values, dtype=float).squeeze().reshape(-1)
    return float(ws_values[1]), float(ti_values[1])


def predict_dataset(
    arrays: Mapping[str, np.ndarray],
    spec: CandidateSpec,
    wake_params: Sequence[float],
    crespo_c0: float,
    crespo_c3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(arrays["t1_ws"])
    ws_pred = np.empty(n, dtype=float)
    ti_pred = np.empty(n, dtype=float)
    for i in range(n):
        ws_pred[i], ti_pred[i] = predict_one(
            arrays, i, spec, wake_params, crespo_c0, crespo_c3
        )
    return ws_pred, ti_pred


class WSObjective:
    def __init__(self, arrays: Mapping[str, np.ndarray], spec: CandidateSpec):
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.spec = spec

    def __call__(self, theta: Sequence[float]) -> float:
        try:
            pred, _ = predict_dataset(
                self.arrays,
                self.spec,
                theta,
                crespo_c0=0.73,
                crespo_c3=-0.32,
            )
            target = self.arrays["t2_ws"]
            rel = (pred - target) / np.maximum(target, 1e-6)
            if not np.all(np.isfinite(rel)):
                return 1e9
            return float(np.mean(rel**2) + 0.25 * np.mean(rel) ** 2)
        except Exception:
            return 1e9


class TIObjective:
    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        spec: CandidateSpec,
        wake_params: Sequence[float],
    ):
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.spec = spec
        self.wake_params = tuple(float(v) for v in wake_params)

    def __call__(self, theta: Sequence[float]) -> float:
        try:
            c0, c3 = map(float, theta)
            _, pred = predict_dataset(
                self.arrays,
                self.spec,
                self.wake_params,
                crespo_c0=c0,
                crespo_c3=c3,
            )
            target = self.arrays["t2_ti_pct"]
            error_pp = pred - target
            if not np.all(np.isfinite(error_pp)):
                return 1e9
            return float(np.mean(error_pp**2) + 0.25 * np.mean(error_pp) ** 2)
        except Exception:
            return 1e9


# -----------------------------------------------------------------------------
# Metrics and reporting
# -----------------------------------------------------------------------------


def ws_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - target
    rel = 100.0 * error / np.maximum(np.abs(target), 1e-9)
    return {
        "n": int(len(target)),
        "bias_mps": float(np.mean(error)),
        "mae_mps": float(np.mean(np.abs(error))),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "mean_abs_relative_error_pct": float(np.mean(np.abs(rel))),
        "p95_abs_relative_error_pct": float(np.percentile(np.abs(rel), 95)),
        "max_abs_relative_error_pct": float(np.max(np.abs(rel))),
    }


def ti_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    error = pred - target
    rel = 100.0 * error / np.maximum(np.abs(target), 1e-9)
    return {
        "n": int(len(target)),
        "bias_percentage_points": float(np.mean(error)),
        "mae_percentage_points": float(np.mean(np.abs(error))),
        "rmse_percentage_points": float(np.sqrt(np.mean(error**2))),
        "mean_abs_relative_error_pct": float(np.mean(np.abs(rel))),
        "p95_abs_error_percentage_points": float(np.percentile(np.abs(error), 95)),
        "max_abs_error_percentage_points": float(np.max(np.abs(error))),
    }


def save_parity_plot(
    target: np.ndarray,
    pred: np.ndarray,
    split: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    for name in ["train", "validation"]:
        mask = split == name
        if np.any(mask):
            ax.scatter(target[mask], pred[mask], s=18, alpha=0.7, label=name)
    low = float(min(np.nanmin(target), np.nanmin(pred)))
    high = float(max(np.nanmax(target), np.nanmax(pred)))
    margin = 0.03 * max(high - low, 1e-6)
    ax.plot([low - margin, high + margin], [low - margin, high + margin], "--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_residual_spacing_plot(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(df["spacing_D"], df["ws_error_pct"], s=18, alpha=0.65)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("T1-T2 spacing [D]")
    ax.set_ylabel("T2 effective wind-speed error [%]")
    ax.set_title("Calibrated PyWake residual versus spacing")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def flatten_metrics(rows: List[Tuple[str, str, Mapping[str, float]]]) -> pd.DataFrame:
    records = []
    for quantity, split, metrics in rows:
        record = {"quantity": quantity, "split": split}
        record.update(metrics)
        records.append(record)
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Calibration driver
# -----------------------------------------------------------------------------


def wake_bounds(model: str) -> List[Tuple[float, float]]:
    if model == "bastankhah":
        return [
            (0.015, 0.120),  # k
            (0.05, 0.50),    # ceps
            (1.0, 1.0),      # Jimenez scale fixed: neutral cases contain no yaw information
        ]
    if model == "zong":
        return [
            (0.015, 0.120),  # effective constant k via a=[0,k]
            (0.15, 0.60),    # eps_coeff
            (1.0, 1.0),      # Jimenez scale fixed: neutral cases contain no yaw information
        ]
    raise ValueError(model)


def parameter_names(model: str) -> List[str]:
    if model == "bastankhah":
        return ["wake_k", "ceps", "deflection_scale"]
    if model == "zong":
        return ["wake_k_effective", "eps_coeff", "deflection_scale"]
    raise ValueError(model)


def optimize_ws_candidate(
    train_arrays: Mapping[str, np.ndarray],
    val_arrays: Mapping[str, np.ndarray],
    spec: CandidateSpec,
    *,
    maxiter: int,
    popsize: int,
    workers: int,
    random_seed: int,
) -> Dict[str, Any]:
    print(f"\n[WS calibration] Candidate: {spec.name}", flush=True)
    bounds = wake_bounds(spec.model)
    initial = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)

    # Explicit smoke test gives a useful error instead of an opaque 1e9 objective.
    try:
        predict_one(train_arrays, 0, spec, initial, 0.73, -0.32)
    except Exception as exc:
        raise RuntimeError(f"Candidate {spec.name} failed smoke test: {exc}") from exc

    objective = WSObjective(train_arrays, spec)
    start = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(random_seed),
        polish=True,
        workers=max(1, int(workers)),
        updating="deferred" if workers > 1 else "immediate",
        disp=True,
    )
    seconds = time.perf_counter() - start
    params = np.asarray(result.x, dtype=float)
    train_pred, _ = predict_dataset(train_arrays, spec, params, 0.73, -0.32)
    val_pred, _ = predict_dataset(val_arrays, spec, params, 0.73, -0.32)
    train_metric = ws_metrics(train_arrays["t2_ws"], train_pred)
    val_metric = ws_metrics(val_arrays["t2_ws"], val_pred)
    print(
        f"  train MARE={train_metric['mean_abs_relative_error_pct']:.3f}% | "
        f"validation MARE={val_metric['mean_abs_relative_error_pct']:.3f}%",
        flush=True,
    )
    return {
        "candidate": spec.name,
        "spec": asdict(spec),
        "wake_parameter_names": parameter_names(spec.model),
        "wake_parameters": params.tolist(),
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "seconds": float(seconds),
        "train_ws_metrics": train_metric,
        "validation_ws_metrics": val_metric,
    }


def optimize_ti(
    train_arrays: Mapping[str, np.ndarray],
    val_arrays: Mapping[str, np.ndarray],
    spec: CandidateSpec,
    wake_params: Sequence[float],
    *,
    maxiter: int,
    popsize: int,
    workers: int,
    random_seed: int,
) -> Dict[str, Any]:
    print(f"\n[TI calibration] Candidate: {spec.name}", flush=True)
    bounds = [(0.10, 2.00), (-1.00, -0.05)]
    objective = TIObjective(train_arrays, spec, wake_params)
    start = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=int(maxiter),
        popsize=int(popsize),
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(random_seed + 1000),
        polish=True,
        workers=max(1, int(workers)),
        updating="deferred" if workers > 1 else "immediate",
        disp=True,
    )
    seconds = time.perf_counter() - start
    c0, c3 = map(float, result.x)
    _, train_pred = predict_dataset(train_arrays, spec, wake_params, c0, c3)
    _, val_pred = predict_dataset(val_arrays, spec, wake_params, c0, c3)
    train_metric = ti_metrics(train_arrays["t2_ti_pct"], train_pred)
    val_metric = ti_metrics(val_arrays["t2_ti_pct"], val_pred)
    print(
        f"  train TI MAE={train_metric['mae_percentage_points']:.3f} pp | "
        f"validation TI MAE={val_metric['mae_percentage_points']:.3f} pp",
        flush=True,
    )
    return {
        "crespo_c": [c0, 0.8325, 0.0325, c3],
        "objective": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "seconds": float(seconds),
        "train_ti_metrics": train_metric,
        "validation_ti_metrics": val_metric,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Six-seed-averaged FAST.Farm calibration file (.csv, .xlsx, or .xls)",
    )
    parser.add_argument("--pitch-table", type=Path, default=Path("data/pitch_ct_tilt_map.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("caliberation/results/t2_pywake_calibration"))
    parser.add_argument("--models", nargs="+", default=["bastankhah", "zong"], choices=["bastankhah", "zong"])
    parser.add_argument("--rotor-models", nargs="+", default=["gq_power3", "gq"], choices=["center", "gq", "gq_power3", "gaussian_overlap"])
    parser.add_argument("--yaw-signs", nargs="+", type=int, default=[-1, 1], choices=[-1, 1])
    parser.add_argument("--tilt-mode", default="six_plus_csv", choices=["zero", "fixed6", "fixed", "csv", "csv_platform", "csv_only", "six_plus_csv", "six_minus_csv"])
    parser.add_argument("--fixed-tilt-deg", type=float, default=6.0)
    parser.add_argument("--tilt-iterations", type=int, default=2)
    parser.add_argument("--diameter", type=float, default=240.0)
    parser.add_argument("--hub-height", type=float, default=150.0)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("--wd-mode", choices=["fixed", "metadata"], default="fixed")
    parser.add_argument("--fixed-wd", type=float, default=270.0)
    parser.add_argument("--average-sea-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=20260708)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", "1")))
    parser.add_argument("--maxiter-ws", type=int, default=16)
    parser.add_argument("--popsize-ws", type=int, default=6)
    parser.add_argument("--maxiter-ti", type=int, default=14)
    parser.add_argument("--popsize-ti", type=int, default=6)
    parser.add_argument(
        "--compare-all-rotor-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "By default, use the first working rotor model in --rotor-models "
            "for each wake-model/yaw-sign pair. Enable this to optimize every "
            "listed rotor model, which is substantially more expensive."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test limit; keeps whole physical groups")
    args = parser.parse_args()

    root = Path.cwd()
    if args.data is None:
        data_dir = root / "caliberation" / "data"
    
        candidates = sorted(
            p
            for pattern in ("*.xlsx", "*.xls", "*.csv")
            for p in data_dir.glob(pattern)
            if "pitch_ct" not in p.name.lower()
        )
    
        if len(candidates) != 1:
            raise RuntimeError(
                "--data was not supplied and automatic detection did not find exactly "
                "one non-pitch calibration file (.csv, .xlsx, or .xls) in "
                f"{data_dir}. Found: {candidates}"
            )
    
        args.data = candidates[0]
    args.data = args.data.resolve()
    args.pitch_table = args.pitch_table.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PYWAKE T1 -> T2 FAST.FARM CALIBRATION")
    print("=" * 78)
    print(f"PyWake version : {getattr(py_wake, '__version__', 'unknown')}")
    print(f"Python         : {sys.executable}")
    print(f"Calibration data: {args.data}")
    print(f"Pitch/Ct table : {args.pitch_table}")
    print(f"Output         : {args.output_dir}")
    print(f"Workers        : {args.workers}")

    data, column_mapping, split_keys = load_and_prepare_data(
        args.data,
        average_sea_state=bool(args.average_sea_state),
        wd_mode=args.wd_mode,
        fixed_wd=args.fixed_wd,
        max_rows=args.max_rows,
        random_seed=args.random_seed,
    )
    data = split_train_validation(data, args.validation_fraction, args.random_seed)
    train = data[data.split == "train"].reset_index(drop=True)
    validation = data[data.split == "validation"].reset_index(drop=True)
    train_arrays = dataframe_to_arrays(train)
    val_arrays = dataframe_to_arrays(validation)

    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "pywake_version": getattr(py_wake, "__version__", "unknown"),
        "bastankhah_signature": str(inspect.signature(Bastankhah_PorteAgel_2014)),
        "zong_signature": str(inspect.signature(Zong_PorteAgel_2020)) if Zong_PorteAgel_2020 else None,
        "crespo_signature": str(inspect.signature(CrespoHernandez)),
        "arguments": vars(args),
        "data_sha256": file_sha256(args.data),
        "pitch_table_sha256": file_sha256(args.pitch_table),
        "column_mapping": column_mapping,
        "physical_split_keys": split_keys,
    }
    atomic_write_json(args.output_dir / "environment_and_inputs.json", environment)

    candidate_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for model in args.models:
        for yaw_sign in args.yaw_signs:
            successful_rotor_for_pair = False
            for rotor_model in args.rotor_models:
                spec = CandidateSpec(
                    model=model,
                    rotor_model=rotor_model,
                    yaw_sign=int(yaw_sign),
                    tilt_mode=args.tilt_mode,
                    fixed_tilt_deg=float(args.fixed_tilt_deg),
                    tilt_iterations=int(args.tilt_iterations),
                    diameter=float(args.diameter),
                    hub_height=float(args.hub_height),
                    rho=float(args.rho),
                    pitch_table=str(args.pitch_table),
                )
                try:
                    result = optimize_ws_candidate(
                        train_arrays,
                        val_arrays,
                        spec,
                        maxiter=args.maxiter_ws,
                        popsize=args.popsize_ws,
                        workers=args.workers,
                        random_seed=args.random_seed,
                    )
                    candidate_results.append(result)
                    atomic_write_json(
                        args.output_dir / f"candidate_{spec.name}.json",
                        result,
                    )
                    successful_rotor_for_pair = True
                    if not args.compare_all_rotor_models:
                        break
                except Exception as exc:
                    print(f"[candidate failed] {spec.name}: {exc}", file=sys.stderr, flush=True)
                    failures.append({
                        "candidate": spec.name,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
            if not successful_rotor_for_pair:
                print(
                    f"[pair failed] model={model}, yaw_sign={yaw_sign}: no rotor model worked",
                    file=sys.stderr,
                    flush=True,
                )

    atomic_write_json(args.output_dir / "candidate_failures.json", {"failures": failures})
    if not candidate_results:
        raise RuntimeError("All candidate models failed. See candidate_failures.json")

    candidate_results.sort(
        key=lambda r: r["validation_ws_metrics"]["mean_abs_relative_error_pct"]
    )
    best_ws = candidate_results[0]
    best_spec = CandidateSpec(**best_ws["spec"])
    best_wake_params = best_ws["wake_parameters"]

    summary_rows = []
    for rank, result in enumerate(candidate_results, start=1):
        row = {
            "rank": rank,
            "candidate": result["candidate"],
            "objective": result["objective"],
            "seconds": result["seconds"],
            "success": result["success"],
        }
        for name, value in zip(result["wake_parameter_names"], result["wake_parameters"]):
            row[name] = value
        row.update({f"train_{k}": v for k, v in result["train_ws_metrics"].items()})
        row.update({f"validation_{k}": v for k, v in result["validation_ws_metrics"].items()})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "ws_candidate_summary.csv", index=False)

    ti_result = optimize_ti(
        train_arrays,
        val_arrays,
        best_spec,
        best_wake_params,
        maxiter=args.maxiter_ti,
        popsize=args.popsize_ti,
        workers=args.workers,
        random_seed=args.random_seed,
    )
    crespo_c = ti_result["crespo_c"]

    all_arrays = dataframe_to_arrays(data)
    ws_pred, ti_pred = predict_dataset(
        all_arrays,
        best_spec,
        best_wake_params,
        crespo_c0=crespo_c[0],
        crespo_c3=crespo_c[3],
    )
    predictions = data.copy()
    predictions["pywake_t2_ws_mps"] = ws_pred
    predictions["pywake_t2_ti_pct"] = ti_pred
    predictions["ws_error_mps"] = ws_pred - predictions["t2_ws"]
    predictions["ws_error_pct"] = 100.0 * predictions["ws_error_mps"] / predictions["t2_ws"]
    predictions["ti_error_percentage_points"] = ti_pred - predictions["t2_ti_pct"]
    predictions.to_csv(args.output_dir / "calibrated_predictions_all.csv", index=False)

    metric_rows: List[Tuple[str, str, Mapping[str, float]]] = []
    for split in ["train", "validation", "all"]:
        mask = np.ones(len(predictions), dtype=bool) if split == "all" else predictions.split.to_numpy() == split
        metric_rows.append(("T2_effective_WS", split, ws_metrics(
            predictions.loc[mask, "t2_ws"].to_numpy(),
            predictions.loc[mask, "pywake_t2_ws_mps"].to_numpy(),
        )))
        metric_rows.append(("T2_effective_TI", split, ti_metrics(
            predictions.loc[mask, "t2_ti_pct"].to_numpy(),
            predictions.loc[mask, "pywake_t2_ti_pct"].to_numpy(),
        )))
    metrics_frame = flatten_metrics(metric_rows)
    metrics_frame.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    by_spacing = []
    spacing_round = predictions["spacing_D"].round(3)
    for spacing, subset in predictions.groupby(spacing_round):
        record = {"spacing_D": float(spacing), "n": len(subset)}
        record.update({f"ws_{k}": v for k, v in ws_metrics(
            subset["t2_ws"].to_numpy(), subset["pywake_t2_ws_mps"].to_numpy()
        ).items()})
        record.update({f"ti_{k}": v for k, v in ti_metrics(
            subset["t2_ti_pct"].to_numpy(), subset["pywake_t2_ti_pct"].to_numpy()
        ).items()})
        by_spacing.append(record)
    pd.DataFrame(by_spacing).to_csv(args.output_dir / "metrics_by_spacing.csv", index=False)

    save_parity_plot(
        predictions["t2_ws"].to_numpy(),
        predictions["pywake_t2_ws_mps"].to_numpy(),
        predictions["split"].to_numpy(),
        "FAST.Farm T2 effective wind speed [m/s]",
        "PyWake T2 effective wind speed [m/s]",
        "T2 effective wind-speed calibration",
        args.output_dir / "parity_t2_effective_ws.png",
    )
    save_parity_plot(
        predictions["t2_ti_pct"].to_numpy(),
        predictions["pywake_t2_ti_pct"].to_numpy(),
        predictions["split"].to_numpy(),
        "FAST.Farm T2 effective TI [%]",
        "PyWake T2 effective TI [%]",
        "T2 effective-TI calibration",
        args.output_dir / "parity_t2_effective_ti.png",
    )
    save_residual_spacing_plot(
        predictions,
        args.output_dir / "ws_residual_vs_spacing.png",
    )

    final_config = {
        "purpose": "T1-to-T2 PyWake calibration for load-surrogate inputs",
        "boundary_condition": {
            "ws": "FAST.Farm T1_Mean_RotorEffectiveSpeed_mps",
            "ti": "FAST.Farm T1_TI_RotorEffective_pct / 100",
        },
        "targets": {
            "ws": "FAST.Farm T2_Mean_RotorEffectiveSpeed_mps",
            "ti": "FAST.Farm T2_TI_RotorEffective_pct",
        },
        "wake_model": best_spec.model,
        "rotor_average_model": best_spec.rotor_model,
        "yaw_sign": best_spec.yaw_sign,
        "tilt_mode": best_spec.tilt_mode,
        "fixed_tilt_deg": best_spec.fixed_tilt_deg,
        "tilt_iterations": best_spec.tilt_iterations,
        "pywake_wd_mode": args.wd_mode,
        "fixed_pywake_wd_deg": args.fixed_wd,
        "wake_parameter_names": best_ws["wake_parameter_names"],
        "wake_parameters": best_wake_params,
        "wake_parameter_dict": dict(zip(best_ws["wake_parameter_names"], best_wake_params)),
        "crespo_c": crespo_c,
        "diameter_m": args.diameter,
        "hub_height_m": args.hub_height,
        "rho_kg_m3": args.rho,
        "average_sea_state": bool(args.average_sea_state),
        "pitch_table": str(args.pitch_table),
        "calibration_data": str(args.data),
        "training_ws_metrics": best_ws["train_ws_metrics"],
        "validation_ws_metrics": best_ws["validation_ws_metrics"],
        "training_ti_metrics": ti_result["train_ti_metrics"],
        "validation_ti_metrics": ti_result["validation_ti_metrics"],
        "files": {
            "predictions": str(args.output_dir / "calibrated_predictions_all.csv"),
            "metrics": str(args.output_dir / "metrics_summary.csv"),
            "candidate_summary": str(args.output_dir / "ws_candidate_summary.csv"),
        },
    }
    atomic_write_json(args.output_dir / "pywake_t2_calibration.json", final_config)
    atomic_write_json(args.output_dir / "ti_optimization.json", ti_result)

    print("\n" + "=" * 78)
    print("CALIBRATION COMPLETE")
    print("=" * 78)
    print(f"Best model       : {best_spec.name}")
    print(f"Wake parameters  : {dict(zip(best_ws['wake_parameter_names'], best_wake_params))}")
    print(f"Crespo c         : {crespo_c}")
    print(
        "Validation WS   : "
        f"MARE={best_ws['validation_ws_metrics']['mean_abs_relative_error_pct']:.3f}%"
    )
    print(
        "Validation TI   : "
        f"MAE={ti_result['validation_ti_metrics']['mae_percentage_points']:.3f} percentage points"
    )
    print(f"Configuration    : {args.output_dir / 'pywake_t2_calibration.json'}")
    print(f"Predictions      : {args.output_dir / 'calibrated_predictions_all.csv'}")


if __name__ == "__main__":
    main()
