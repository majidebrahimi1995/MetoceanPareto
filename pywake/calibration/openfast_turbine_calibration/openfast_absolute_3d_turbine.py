#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import DensityScale, PowerCtNDTabular

def _cube(df, col, ws, pitch, yaw):
    idx = pd.MultiIndex.from_product(
        [ws,pitch,yaw],
        names=["wind_speed","pitch_offset_deg","yaw_deg"])
    s = df.set_index(["wind_speed","pitch_offset_deg","yaw_deg"])[col].reindex(idx)
    if s.isna().any():
        raise RuntimeError(f"Incomplete table for {col}: {int(s.isna().sum())} missing")
    return s.to_numpy(float).reshape(len(ws),len(pitch),len(yaw))

def build_absolute_openfast_3d_turbine(
    table_csv, diameter_m=240.0, hub_height_m=150.0,
    name="IEA15MW_OpenFAST_Absolute3D_TI6"
):
    df = pd.read_csv(Path(table_csv))
    ws = np.sort(df["wind_speed"].unique().astype(float))
    pitch = np.sort(df["pitch_offset_deg"].unique().astype(float))
    yaw = np.sort(df["yaw_deg"].unique().astype(float))
    power = _cube(df,"power_w",ws,pitch,yaw)
    ct = _cube(df,"ct_x_pywake",ws,pitch,yaw)

    pct = PowerCtNDTabular(
        input_keys=["ws","pitch_offset_deg","yaw"],
        value_lst=[ws,pitch,yaw],
        power_arr=power,
        power_unit="w",
        ct_arr=ct,
        additional_models=[DensityScale()],
    )
    wt = WindTurbine(
        name=name,
        diameter=float(diameter_m),
        hub_height=float(hub_height_m),
        powerCtFunction=pct,
    )
    return wt, {"table":df,"ws_grid":ws,"pitch_grid":pitch,"yaw_grid":yaw}
