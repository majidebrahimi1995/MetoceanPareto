#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DEFAULT_STACK = Path(__file__).resolve().parents[2] / "deployment" / "CURRENT"

POWER_SCALE = 0.9849146361095564
RHO = 1.225
DIAMETER_M = 240.0
HUB_HEIGHT_M = 150.0

# Plot labels follow the turbine order returned by the canonical farm layout.
# Rotation changes x/y coordinates only; it does not change this ordering.
FLOW_MAP_TURBINE_LABELS = {
    "rectangle": ("T1", "T2", "T3", "T4", "T5"),
    "tampen": ("HY01", "HY02", "HY03", "HY10", "HY11"),
}


def parse_vector(text: str | None, n: int, name: str) -> np.ndarray:
    if text is None:
        return np.zeros(n, dtype=float)
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) == 1:
        vals *= n
    if len(vals) != n:
        raise ValueError(
            f"{name} must contain one value or exactly {n} comma-separated values; got {len(vals)}"
        )
    return np.asarray(vals, dtype=float)


def extract(sim, name: str) -> np.ndarray:
    obj = getattr(sim, name, None)
    if obj is None:
        try:
            obj = sim[name]
        except Exception:
            obj = None
    if obj is None:
        raise KeyError(f"PyWake result has no field {name!r}")
    return np.asarray(getattr(obj, "values", obj), dtype=float).squeeze().reshape(-1)


def make_ctn_interpolator(table_path: Path):
    with table_path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    ws = np.array(sorted({float(r["wind_speed"]) for r in rows}), dtype=float)
    pitch = np.array(sorted({float(r["pitch_offset_deg"]) for r in rows}), dtype=float)
    yaw = np.array(sorted({float(r["yaw_deg"]) for r in rows}), dtype=float)

    iw = {v: i for i, v in enumerate(ws)}
    ip = {v: i for i, v in enumerate(pitch)}
    iy = {v: i for i, v in enumerate(yaw)}

    arr = np.full((len(ws), len(pitch), len(yaw)), np.nan, dtype=float)
    for r in rows:
        arr[
            iw[float(r["wind_speed"])],
            ip[float(r["pitch_offset_deg"])],
            iy[float(r["yaw_deg"])],
        ] = float(r["ct_n_openfast"])

    if np.isnan(arr).any():
        raise RuntimeError("ct_n_openfast table is incomplete")

    interp = RegularGridInterpolator(
        (ws, pitch, yaw), arr, method="linear", bounds_error=True
    )
    return interp, ws, pitch, yaw



class PyWakeDomainError(ValueError):
    """Candidate is outside the canonical turbine-table domain."""


def _vector(value, n, name):
    if value is None or isinstance(value, str):
        result = parse_vector(value, n, name)
    else:
        result = np.asarray(value, dtype=float).reshape(-1)
        if result.size == 1:
            result = np.repeat(result, n)
    if result.size != n or not np.isfinite(result).all():
        raise PyWakeDomainError(f"{name} must contain {n} finite values")
    return result


class FrozenPyWakeEvaluator:
    """Persistent interface owned by this canonical runner.

    CLI and connected optimizer use this exact calculation. Runtime geometry,
    calibration, turbine identity and physical-WD conversion remain here.
    Instances are process-local and must not be shared between threads.
    """

    def __init__(self, stack=DEFAULT_STACK):
        stack = Path(stack).resolve()
        calib_json = stack / "pywake" / "pywake_absolute3d_calibration.json"
        table3d = stack / "pywake" / "openfast_absolute_3d_pywake_table_TI6_PROJECTED_CT_BOUNDED.csv"
        old_pitch = stack / "pywake" / "pitch_ct_tilt_map.csv"
        runtime = stack / "runtime"

        for p in [calib_json, table3d, old_pitch, runtime]:
            if not p.exists():
                raise FileNotFoundError(p)

        sys.path.insert(0, str(runtime))

        from rotation_convention import (
            layout_rotation_from_physical_wd,
            canonical_state_from_rotation,
        )
        import calibrate_pywake_t2 as c
        from openfast_absolute_3d_turbine import build_absolute_openfast_3d_turbine
        from py_wake.site import UniformSite


        cfg = json.loads(calib_json.read_text(encoding="utf-8"))
        wake = np.asarray(cfg["wake_parameters"], dtype=float)
        crespo = np.asarray(cfg["crespo_c"], dtype=float)

        turbine3d, table_info = build_absolute_openfast_3d_turbine(
            table3d,
            diameter_m=DIAMETER_M,
            hub_height_m=HUB_HEIGHT_M,
        )

        ctn_interp, _, ctn_pitch, ctn_yaw = make_ctn_interpolator(table3d)


        original_builder = c.build_pitch_table_model_cached
        old_model = original_builder(
            str(old_pitch),
            DIAMETER_M,
            HUB_HEIGHT_M,
            RHO,
        )

        composite = c.PitchTableModel(
            turbine=turbine3d,
            table=old_model.table,
            ws_grid=np.asarray(old_model.ws_grid, float),
            pitch_grid=np.asarray(old_model.pitch_grid, float),
            platform_pitch_table=old_model.platform_pitch_table,
        )

        def patched_builder(*_a, **_kw):
            return composite

        spec = c.CandidateSpec(
            model="zong",
            rotor_model="gaussian_overlap",
            yaw_sign=1,
            tilt_mode="six_plus_csv",
            fixed_tilt_deg=6.0,
            tilt_iterations=2,
            diameter=DIAMETER_M,
            hub_height=HUB_HEIGHT_M,
            rho=RHO,
            pitch_table=str(old_pitch),
        )


        self.stack = stack
        self._c = c
        self._rotation = layout_rotation_from_physical_wd
        self._state = canonical_state_from_rotation
        self._UniformSite = UniformSite
        self._wake, self._crespo = wake, crespo
        self._composite, self._spec = composite, spec
        self._ctn_interp = ctn_interp
        self._ctn_pitch, self._ctn_yaw = ctn_pitch, ctn_yaw
        self._table_info = table_info
        self._models = {}

    def evaluate(
        self,
        *,
        farm,
        ws,
        wd,
        ti,
        yaw=None,
        pitch=None,
        return_context=False,
    ):
        args = SimpleNamespace(farm=farm, ws=float(ws), wd=float(wd), ti=float(ti),
                               yaw=yaw, pitch=pitch)
        if not np.isfinite([args.ws, args.wd, args.ti]).all() or args.ws <= 0 or args.ti <= 0:
            raise PyWakeDomainError("Ambient inputs must be finite and WS/TI positive")
        c = self._c
        layout_rotation_from_physical_wd, canonical_state_from_rotation = self._rotation, self._state
        composite, spec = self._composite, self._spec
        ctn_interp, ctn_pitch, ctn_yaw = self._ctn_interp, self._ctn_pitch, self._ctn_yaw
        table_info = self._table_info
        rotation = layout_rotation_from_physical_wd(args.wd)
        state = canonical_state_from_rotation(
            farm=args.farm,
            layout_rotation_deg=rotation,
        )

        x = np.asarray(state["x_m"], dtype=float)
        y = np.asarray(state["y_m"], dtype=float)
        turbines = list(state["turbines"])
        n = len(turbines)

        yaw = _vector(args.yaw, n, "yaw")
        pitch = _vector(args.pitch, n, "pitch")


        if np.any(yaw < ctn_yaw[0]) or np.any(yaw > ctn_yaw[-1]):
            raise PyWakeDomainError(f"Yaw outside frozen table range [{ctn_yaw[0]}, {ctn_yaw[-1]}] deg")
        if np.any(pitch < ctn_pitch[0]) or np.any(pitch > ctn_pitch[-1]):
            raise PyWakeDomainError(f"Pitch outside frozen table range [{ctn_pitch[0]}, {ctn_pitch[-1]}] deg")

        ws_grid = np.asarray(table_info["ws_grid"], dtype=float)
        if not (ws_grid[0] <= args.ws <= ws_grid[-1]):
            raise PyWakeDomainError(
                f"Ambient WS={args.ws} outside frozen PyWake support [{ws_grid[0]}, {ws_grid[-1]}] m/s"
            )


        key = float(args.ti)
        if key not in self._models:
            UniformSite, wake, crespo = self._UniformSite, self._wake, self._crespo
            original = c.build_pitch_table_model_cached
            c.build_pitch_table_model_cached = lambda *_a, **_kw: composite
            try:
                site = UniformSite(p_wd=[1.0], ti=args.ti / 100.0)
            
                wfm = c.build_wfm(
                    spec,
                    site,
                    wake,
                    crespo_c0=float(crespo[0]),
                    crespo_c3=float(crespo[3]),
                )
                self._models[key] = wfm
            finally:
                c.build_pitch_table_model_cached = original
        wfm = self._models[key]
        yaw_ilk = yaw[:, None, None]
        pitch_ilk = pitch[:, None, None]

        ws_guess = np.full(n, args.ws, dtype=float)
        platform_pitch = np.zeros(n)
        rotor_tilt = np.zeros(n)
        sim = None

        for _ in range(max(1, int(spec.tilt_iterations))):
            platform_pitch = composite.platform_pitch(ws_guess, pitch)
            rotor_tilt = c.compute_tilt(
                composite,
                ws_guess,
                pitch,
                spec.tilt_mode,
                spec.fixed_tilt_deg,
            )
            sim = wfm(
                x,
                y,
                wd=[float(state["pywake_wd_deg"])],
                ws=[args.ws],
                yaw=yaw_ilk,
                tilt=rotor_tilt[:, None, None],
                pitch_offset_deg=pitch_ilk,
            )
            ws_guess = extract(sim, "WS_eff")

        ws_eff = extract(sim, "WS_eff")
        ti_eff = 100.0 * extract(sim, "TI_eff")
        power_raw_kw = extract(sim, "Power") / 1000.0
        power_kw = POWER_SCALE * power_raw_kw

        try:
            ct_x = extract(sim, "CT")
        except KeyError:
            ct_x = np.asarray(
                composite.turbine.ct(ws_eff, yaw=yaw, pitch_offset_deg=pitch),
                dtype=float,
            ).reshape(-1)

        pts = np.column_stack([ws_eff, pitch, yaw])
        ct_n = np.asarray(ctn_interp(pts), dtype=float).reshape(-1)

        wd_phys = float(state["physical_wd_deg"])
        wd_rad = math.radians(wd_phys)
        wd_sin = math.sin(wd_rad)
        wd_cos = math.cos(wd_rad)

        rows = []
        for j, t in enumerate(turbines):
            rows.append({
                "Farm": args.farm,
                "Turbine": int(t),
                "Ambient_WS_mps": float(args.ws),
                "Ambient_TI_pct": float(args.ti),
                "Physical_WD_deg": wd_phys,
                "LayoutRotation_deg": float(state["layout_rotation_deg"]),
                "PyWake_WD_deg": float(state["pywake_wd_deg"]),
                "x_m": float(x[j]),
                "y_m": float(y[j]),
                "YawCmd_deg": float(yaw[j]),
                "PitchCmd_deg": float(pitch[j]),
                "WS_eff_mps": float(ws_eff[j]),
                "TI_eff_pct": float(ti_eff[j]),
                "Power_kW": float(power_kw[j]),
                "Ct_n": float(ct_n[j]),
                "Ct_x": float(ct_x[j]),
                "Surrogate_WD_deg": wd_phys,
                "Surrogate_WD_sin": wd_sin,
                "Surrogate_WD_cos": wd_cos,
                "PlatformPitch_deg": float(platform_pitch[j]),
                "RotorTilt_deg": float(rotor_tilt[j]),
            })
        if return_context:
            return rows, {
                "sim": sim,
                "x": x,
                "y": y,
                "state": state,
                "yaw": yaw,
                "pitch": pitch,
                "rotor_tilt": rotor_tilt,
            }

        return rows


def save_velocity_flow_map(
    *,
    context,
    farm,
    ws,
    wd,
    ti,
    png_path,
    pdf_path,
    resolution,
    extend,
):
    """Save the frozen PyWake hub-height effective-velocity field.

    The SimulationResult in ``context`` is the exact result produced by the
    canonical calibrated evaluator, including the physical-WD rotation,
    turbine-specific yaw, pitch-dependent 3-D turbine table, platform pitch,
    and rotor tilt.  No second or generic PyWake model is constructed here.
    """
    if resolution < 50:
        raise ValueError("flow-map resolution must be at least 50")
    if not (0.0 <= extend <= 2.0):
        raise ValueError("flow-map extend must lie in [0, 2]")

    # Import plotting dependencies only when a map is requested so the normal
    # tabular evaluator retains its original lightweight behaviour.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from py_wake.flow_map import HorizontalGrid

    png_path = Path(png_path).expanduser().resolve()
    pdf_path = Path(pdf_path).expanduser().resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    grid = HorizontalGrid(
        resolution=int(resolution),
        extend=float(extend),
        h=HUB_HEIGHT_M,
    )
    flow = context["sim"].flow_map(grid)

    fig, ax = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    plt.sca(ax)
    flow.plot_wake_map()

    # PyWake normally draws turbine locations itself.  These markers ensure
    # that the exact rotated farm coordinates remain visible in all versions.
    ax.scatter(
        context["x"],
        context["y"],
        marker="^",
        s=30,
        facecolor="black",
        edgecolor="white",
        linewidth=0.4,
        zorder=10,
    )

    farm_key = str(farm).strip().casefold()
    labels = FLOW_MAP_TURBINE_LABELS.get(farm_key)
    if labels is None:
        labels = tuple(f"T{i + 1}" for i in range(len(context["x"])))
    if len(labels) != len(context["x"]):
        raise ValueError(
            f"Flow-map label count for {farm!r} is {len(labels)}, but the "
            f"layout contains {len(context['x'])} turbines"
        )

    # Offset in display points keeps labels readable at every map extent and
    # resolution.  The white box prevents wake contours from hiding the text.
    for x_m, y_m, label in zip(context["x"], context["y"], labels):
        ax.annotate(
            label,
            xy=(float(x_m), float(y_m)),
            xytext=(6, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=11,
        )
    ax.set_title(
        f"{farm}: hub-height effective velocity; "
        f"WS={ws:g} m/s, WD={wd:g} deg, TI={ti:g}%"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path

def main():
    ap = argparse.ArgumentParser(
        description="Run arbitrary WS/WD/TI/yaw/pitch through the frozen PyWake model."
    )
    ap.add_argument("--farm", required=True, choices=["Tampen", "Rectangle"])
    ap.add_argument("--ws", required=True, type=float, help="Ambient wind speed [m/s]")
    ap.add_argument("--wd", required=True, type=float, help="Physical meteorological WD [deg]")
    ap.add_argument("--ti", required=True, type=float, help="Ambient turbulence intensity [%%]")
    ap.add_argument("--yaw", default=None, help="One value or N comma-separated yaw commands [deg]")
    ap.add_argument("--pitch", default=None, help="One value or N comma-separated pitch offsets [deg]")
    ap.add_argument("--stack", type=Path, default=DEFAULT_STACK)
    ap.add_argument("--output", type=Path, default=None, help="Optional output CSV")
    ap.add_argument(
        "--flow-map",
        type=Path,
        default=None,
        help=(
            "Optional velocity-contour PNG. A vector PDF with the same stem "
            "is also written unless --flow-map-pdf is supplied."
        ),
    )
    ap.add_argument(
        "--flow-map-pdf",
        type=Path,
        default=None,
        help="Optional explicit vector-PDF path for the velocity contour",
    )
    ap.add_argument(
        "--flow-map-resolution",
        type=int,
        default=500,
        help="Horizontal flow-map resolution (default: 500)",
    )
    ap.add_argument(
        "--flow-map-extend",
        type=float,
        default=0.25,
        help="Fractional map extension around the farm (default: 0.25)",
    )
    args = ap.parse_args()

    evaluator = FrozenPyWakeEvaluator(args.stack)
    want_flow_map = args.flow_map is not None or args.flow_map_pdf is not None
    result = evaluator.evaluate(
        farm=args.farm,
        ws=args.ws,
        wd=args.wd,
        ti=args.ti,
        yaw=args.yaw,
        pitch=args.pitch,
        return_context=want_flow_map,
    )
    if want_flow_map:
        rows, context = result
    else:
        rows = result
        context = None
    stack = evaluator.stack
    state = {"layout_rotation_deg": rows[0]["LayoutRotation_deg"],
             "pywake_wd_deg": rows[0]["PyWake_WD_deg"]}
    wd_phys = rows[0]["Physical_WD_deg"]
    wd_sin, wd_cos = rows[0]["Surrogate_WD_sin"], rows[0]["Surrogate_WD_cos"]
    power_kw = np.asarray([row["Power_kW"] for row in rows])
    print("=" * 100)
    print("FROZEN PYWAKE CASE")
    print("=" * 100)
    print(f"Stack           : {stack}")
    print(f"Farm            : {args.farm}")
    print(f"Ambient WS      : {args.ws:.4f} m/s")
    print(f"Ambient TI      : {args.ti:.4f} %")
    print(f"Physical WD     : {wd_phys:.4f} deg")
    print(f"Layout rotation : {float(state['layout_rotation_deg']):+.4f} deg")
    print(f"PyWake WD       : {float(state['pywake_wd_deg']):.4f} deg")
    print(f"Farm power      : {np.sum(power_kw)/1000.0:.6f} MW")
    print()
    print(
        f"{'T':>2s} {'x[m]':>11s} {'y[m]':>11s} {'yaw':>8s} {'pitch':>8s} "
        f"{'WS_eff':>10s} {'TI_eff[%]':>10s} {'P[kW]':>11s} {'Ct_n':>9s}"
    )
    for r in rows:
        print(
            f"{r['Turbine']:2d} "
            f"{r['x_m']:11.3f} {r['y_m']:11.3f} "
            f"{r['YawCmd_deg']:8.3f} {r['PitchCmd_deg']:8.3f} "
            f"{r['WS_eff_mps']:10.4f} {r['TI_eff_pct']:10.4f} "
            f"{r['Power_kW']:11.3f} {r['Ct_n']:9.5f}"
        )

    print()
    print("Load-surrogate directional inputs:")
    print(f"  WD_deg = {wd_phys:.6f}")
    print(f"  WD_sin = {wd_sin:.9f}")
    print(f"  WD_cos = {wd_cos:.9f}")

    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out}")

    if want_flow_map:
        if args.flow_map is None:
            flow_png = args.flow_map_pdf.with_suffix(".png")
        else:
            flow_png = args.flow_map
            if flow_png.suffix.lower() != ".png":
                flow_png = flow_png.with_suffix(".png")

        if args.flow_map_pdf is None:
            flow_pdf = flow_png.with_suffix(".pdf")
        else:
            flow_pdf = args.flow_map_pdf
            if flow_pdf.suffix.lower() != ".pdf":
                flow_pdf = flow_pdf.with_suffix(".pdf")

        png_written, pdf_written = save_velocity_flow_map(
            context=context,
            farm=args.farm,
            ws=args.ws,
            wd=wd_phys,
            ti=args.ti,
            png_path=flow_png,
            pdf_path=flow_pdf,
            resolution=args.flow_map_resolution,
            extend=args.flow_map_extend,
        )
        print(f"\nVelocity contour PNG: {png_written}")
        print(f"Velocity contour PDF: {pdf_written}")


if __name__ == "__main__":
    main()

