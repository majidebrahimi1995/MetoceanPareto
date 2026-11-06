from __future__ import annotations

import numpy as np

# Start from the original WindGym farm_configs if available.
try:
    from farm_configs import FARM_LAYOUTS as _ORIGINAL_FARM_LAYOUTS
except Exception:
    _ORIGINAL_FARM_LAYOUTS = {}

FARM_LAYOUTS = {}

# Copy original layouts
for name, layout in _ORIGINAL_FARM_LAYOUTS.items():
    FARM_LAYOUTS[name] = {
        "x": np.asarray(layout["x"], dtype=float),
        "y": np.asarray(layout["y"], dtype=float),
    }

# Make sure the known POLIMI 4T farm exists
FARM_LAYOUTS.setdefault(
    "4t_7d_diagonal",
    {
        "x": np.array([300.0, 1980.0, 3660.0, 5340.0], dtype=float),
        "y": np.array([-210.0, -70.0, 70.0, 210.0], dtype=float),
    },
)

# Example 16T farm, if not already loaded from farm_configs
FARM_LAYOUTS.setdefault(
    "16t_7d",
    {
        "x": np.array(
            [
                0.0, 1680.0, 3360.0, 5040.0,
                0.0, 1680.0, 3360.0, 5040.0,
                0.0, 1680.0, 3360.0, 5040.0,
                0.0, 1680.0, 3360.0, 5040.0,
            ],
            dtype=float,
        ),
        "y": np.array(
            [
                -2100.0, -2940.0, -2100.0, -2940.0,
                -420.0, -1260.0, -420.0, -1260.0,
                1260.0, 420.0, 1260.0, 420.0,
                2940.0, 2100.0, 2940.0, 2100.0,
            ],
            dtype=float,
        ),
    },
)


def add_farm(name: str, x, y) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays/lists")

    if len(x) != len(y):
        raise ValueError(f"x and y length mismatch: {len(x)} != {len(y)}")

    if len(x) == 0:
        raise ValueError("farm must contain at least one turbine")

    FARM_LAYOUTS[name] = {"x": x, "y": y}


def get_farm(name: str) -> dict:
    if name not in FARM_LAYOUTS:
        raise ValueError(
            f"Unknown farm {name!r}. Available farms: {list(FARM_LAYOUTS.keys())}"
        )

    layout = FARM_LAYOUTS[name]
    return {
        "x": np.asarray(layout["x"], dtype=float),
        "y": np.asarray(layout["y"], dtype=float),
    }


# BEGIN USER SIX LAYOUTS

FARM_LAYOUTS.update(
    {
        "polimi_5d": {
            "x": [300.0, 1500.0, 2700.0, 3900.0],
            "y": [-210.0, -70.0, 70.0, 210.0],
        },

        "polimi_7d": {
            "x": [300.0, 1980.0, 3660.0, 5340.0],
            "y": [-210.0, -70.0, 70.0, 210.0],
        },

        "tampen": {
            "x": [
                300.000,
                1758.960,
                3217.920,
                2802.720,
                1343.760,
            ],
            "y": [
                178.120,
                541.020,
                903.920,
                -541.030,
                -903.920,
            ],
        },

        "rectangle": {
            "x": [
                2904.000,
                2904.000,
                300.000,
                300.000,
                1602.000,
            ],
            "y": [
                -1503.416,
                1503.416,
                1503.416,
                -1503.416,
                0.000,
            ],
        },

        "tampen_full": {
            "x": [
                9754.154,
                7863.324,
                5972.493,
                4081.662,
                2190.831,
                300.000,
                9142.143,
                7251.313,
                5360.482,
                3469.651,
                1578.820,
            ],
            "y": [
                -1695.576,
                -1362.171,
                -1028.767,
                -695.362,
                -361.958,
                -28.553,
                361.958,
                695.362,
                1028.767,
                1362.171,
                1695.576,
            ],
        },

        "staggered": {
            "x": [
                3540.0,
                3540.0,
                3540.0,
                3540.0,
                1956.0,
                1956.0,
                1956.0,
                1956.0,
                300.0,
                300.0,
                300.0,
                300.0,
            ],
            "y": [
                -3360.0,
                -1632.0,
                288.0,
                2400.0,
                -2496.0,
                -720.0,
                1296.0,
                3360.0,
                -3168.0,
                -1296.0,
                768.0,
                2928.0,
            ],
        },
    }
)

# END USER SIX LAYOUTS


def list_farms() -> dict:
    return {
        name: {
            "n_turbines": len(layout["x"]),
            "x": np.asarray(layout["x"], dtype=float).tolist(),
            "y": np.asarray(layout["y"], dtype=float).tolist(),
        }
        for name, layout in FARM_LAYOUTS.items()
    }
