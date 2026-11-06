#!/usr/bin/env python3
"""
Canonical rotation convention for the frozen PyWake + load-surrogate stack.

PRODUCTION DEFINITION
---------------------
The public input is:

    layout_rotation_deg

defined as the Cartesian counter-clockwise rotation applied to the
zero-direction coordinates in layouts.txt.

With this definition:

    physical_wd_deg = (-layout_rotation_deg) mod 360

PyWake:
    - rotate the layouts.txt base coordinates by layout_rotation_deg
    - translate/center exactly as done by the FAST.Farm generator
    - use PyWake WD = 270 deg

Load surrogate:
    - use physical_wd_deg directly
    - WD_sin = sin(physical_wd_deg)
    - WD_cos = cos(physical_wd_deg)

DO NOT use the alternative FAST.Farm-generator base in production.
DO NOT add/subtract 90 deg in the production interface.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple
import numpy as np


PYWAKE_WD_DEG = 270.0

BASE_LAYOUTS = {
    "Rectangle": {
        1: (300.000001, -1302.000065, "T1"),
        2: (3306.831072, -1302.000064, "T2"),
        3: (3306.831072, 1302.000065, "T3"),
        4: (300.000000, 1302.000065, "T4"),
        5: (1803.415536, -0.000000, "T5"),
    },
    "Tampen": {
        1: (1382.040000, 1458.960000, "HY01"),
        2: (1744.940000, 0.000000, "HY02"),
        3: (2107.840000, -1458.960000, "HY03"),
        4: (662.889999, -1043.760000, "HY10"),
        5: (300.000000, 415.200000, "HY11"),
    },
}


def wrap360(angle_deg: float) -> float:
    """Map an angle to [0, 360)."""
    return float(angle_deg) % 360.0


def wrap180(angle_deg: float) -> float:
    """Map an angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def layout_rotation_from_physical_wd(physical_wd_deg: float) -> float:
    """
    Rotation of layouts.txt coordinates for a physical meteorological WD.

    Returned in [-180, 180).
    """
    return wrap180(-float(physical_wd_deg))


def physical_wd_from_layout_rotation(layout_rotation_deg: float) -> float:
    """
    Canonical physical/surrogate WD implied by a layouts.txt rotation.
    """
    return wrap360(-float(layout_rotation_deg))


def surrogate_wd_features_from_rotation(layout_rotation_deg: float) -> Dict[str, float]:
    """
    Return the canonical load-surrogate WD and its sin/cos encoding.
    """
    wd = physical_wd_from_layout_rotation(layout_rotation_deg)
    rad = math.radians(wd)
    return {
        "WD_deg": wd,
        "WD_sin": math.sin(rad),
        "WD_cos": math.cos(rad),
    }


def rotate_layout(
    farm: str,
    layout_rotation_deg: float,
    *,
    translate_like_fastfarm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Rotate the layouts.txt base coordinates with standard Cartesian CCW rotation.

    When translate_like_fastfarm=True:
      x -> x - min(x) + 300 m
      y -> y - 0.5*(max(y)+min(y))

    Translation does not affect PyWake wake geometry but reproduces the
    FAST.Farm case-coordinate convention exactly.
    """
    if farm not in BASE_LAYOUTS:
        raise KeyError(f"Unknown farm {farm!r}; available: {list(BASE_LAYOUTS)}")

    turbines = sorted(BASE_LAYOUTS[farm])
    x0 = np.array([BASE_LAYOUTS[farm][t][0] for t in turbines], dtype=float)
    y0 = np.array([BASE_LAYOUTS[farm][t][1] for t in turbines], dtype=float)

    th = math.radians(float(layout_rotation_deg))
    c = math.cos(th)
    s = math.sin(th)

    x = c * x0 - s * y0
    y = s * x0 + c * y0

    if translate_like_fastfarm:
        x = x - np.min(x) + 300.0
        y = y - 0.5 * (np.max(y) + np.min(y))

    return x, y, turbines


def pywake_geometry_from_rotation(
    farm: str,
    layout_rotation_deg: float,
) -> Dict[str, object]:
    """
    Complete canonical PyWake geometry payload from one public rotation input.
    """
    x, y, turbines = rotate_layout(
        farm,
        layout_rotation_deg,
        translate_like_fastfarm=True,
    )
    return {
        "farm": farm,
        "turbines": turbines,
        "x_m": x,
        "y_m": y,
        "pywake_wd_deg": PYWAKE_WD_DEG,
        "layout_rotation_deg": wrap180(layout_rotation_deg),
        "physical_wd_deg": physical_wd_from_layout_rotation(layout_rotation_deg),
    }


def canonical_state_from_rotation(
    farm: str,
    layout_rotation_deg: float,
) -> Dict[str, object]:
    """
    Single-source-of-truth state used by BOTH PyWake and the load surrogate.
    """
    p = pywake_geometry_from_rotation(farm, layout_rotation_deg)
    p["surrogate"] = surrogate_wd_features_from_rotation(layout_rotation_deg)
    return p
