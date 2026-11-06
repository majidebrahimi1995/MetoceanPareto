"""Shared feature map and prediction logic for the isolated OpenFAST DEL models."""

from __future__ import annotations

from typing import Iterable

import numpy as np

TARGETS = (
    "DEL_TowerBase",
    "DEL_BladeRoot",
    "DEL_YawBearing",
    "DEL_FairTenMax",
)
CANONICAL_KEYS = ("windspeed", "TI_percent", "WD_deg", "Yaw_T1", "Pitch_T1")
SEA_COORDS = {
    "H2_T8": (2.0, 8.0),
    "H2_T12": (2.0, 12.0),
    "H4_T8": (4.0, 8.0),
    "H4_T10": (4.0, 10.0),
    "H4_T12": (4.0, 12.0),
    "H6_T8": (6.0, 8.0),
    "H6_T12": (6.0, 12.0),
}


def _broadcast(*values):
    arrays = np.broadcast_arrays(*(np.asarray(v) for v in values))
    return [a.reshape(-1) for a in arrays]


def features(ws, ti, wd, yaw, pitch, sea):
    """Exact 120-degree/reflective WD feature map encoded by the source table.

    cosine harmonics are continuous with zero directional slope at the
    reflection points, unlike applying a sawtooth WD_eq directly.
    """
    ws, ti, wd, yaw, pitch, sea = _broadcast(ws, ti, wd, yaw, pitch, sea)
    sea = np.asarray(sea, dtype=str)
    if np.any(~np.isin(sea, ("Mild", "Normal"))):
        raise ValueError("sea must be Mild or Normal")
    wd = np.asarray(wd, dtype=float)
    folded = np.minimum(np.mod(wd, 120.0), 120.0 - np.mod(wd, 120.0))
    phase = np.pi * folded / 60.0
    return np.column_stack(
        (
            (np.asarray(ws, dtype=float) - 6.5) / 2.5,
            (np.asarray(ti, dtype=float) - 11.0) / 7.0,
            np.cos(phase),
            np.cos(2.0 * phase),
            np.cos(3.0 * phase),
            np.asarray(yaw, dtype=float) / 25.0,
            (np.asarray(pitch, dtype=float) - 2.0) / 2.0,
            np.where(sea == "Normal", 1.0, -1.0),
        )
    ).astype(np.float64, copy=False)


def sea_features(x, hs, tp):
    hs, tp = _broadcast(hs, tp)
    if len(x) != len(hs):
        raise ValueError("Sea inputs and turbine inputs have incompatible lengths")
    return np.column_stack((x, (hs - 4.0) / 2.0, (tp - 10.0) / 2.0))


def _domain_check(pack, ws, ti, yaw, pitch, allow_extrapolation=False):
    if allow_extrapolation:
        return
    values = _broadcast(ws, ti, yaw, pitch)
    for name, v in zip(("ws", "ti", "yaw", "pitch"), values):
        lo, hi = pack["bounds"][name]
        if np.any(np.asarray(v, dtype=float) < lo - 1e-5) or np.any(
            np.asarray(v, dtype=float) > hi + 1e-5
        ):
            raise ValueError(
                f"{name} outside training range [{lo:.4g}, {hi:.4g}]; "
                "supply additional training simulations or explicitly allow extrapolation"
            )


def predict_base(pack, ws, ti, wd, yaw, pitch, sea, allow_extrapolation=False):
    _domain_check(pack, ws, ti, yaw, pitch, allow_extrapolation)
    x = features(ws, ti, wd, yaw, pitch, sea)
    ylog = pack["regressor"].predict(x) * pack["y_scale"] + pack["y_center"]
    return np.exp(ylog)


def predict_continuous_sea(
    base_pack, sea_pack, ws, ti, wd, yaw, pitch, hs, tp,
    allow_extrapolation=False,
):
    ws, ti, wd, yaw, pitch, hs, tp = _broadcast(ws, ti, wd, yaw, pitch, hs, tp)
    if not allow_extrapolation and (
        np.any((hs < 2.0) | (hs > 6.0)) or np.any((tp < 8.0) | (tp > 12.0))
    ):
        raise ValueError("Hs/Tp outside the studied [2,6] m x [8,12] s rectangle")
    b0 = predict_base(base_pack, ws, ti, wd, yaw, pitch, "Mild", allow_extrapolation)
    b1 = predict_base(base_pack, ws, ti, wd, yaw, pitch, "Normal", allow_extrapolation)
    w = (hs - 2.0) / 4.0
    x = features(ws, ti, wd, yaw, pitch, "Mild")
    xx = sea_features(x, hs, tp)
    r = sea_pack["regressor"].predict(xx) * sea_pack["y_scale"] + sea_pack["y_center"]
    x0 = sea_features(x, np.full(len(x), 2.0), np.full(len(x), 8.0))
    x1 = sea_features(x, np.full(len(x), 6.0), np.full(len(x), 12.0))
    r0 = sea_pack["regressor"].predict(x0) * sea_pack["y_scale"] + sea_pack["y_center"]
    r1 = sea_pack["regressor"].predict(x1) * sea_pack["y_scale"] + sea_pack["y_center"]
    corrected = r - (1.0 - w[:, None]) * r0 - w[:, None] * r1
    return np.exp((1.0 - w[:, None]) * np.log(b0) + w[:, None] * np.log(b1) + corrected)


def parse_sea_name(values: Iterable[str]):
    result = []
    for value in values:
        key = str(value)
        if key not in SEA_COORDS:
            raise ValueError(f"Unknown sea state {key!r}; expected one of {sorted(SEA_COORDS)}")
        result.append(SEA_COORDS[key])
    return np.asarray(result, dtype=float)
