#!/usr/bin/env python3
"""Extract a 0--10% epsilon-constraint database from completed searches.

No model fitting or optimization is performed here.  The script reads the
three completed NSGA-II/DE archives for one layout/centroid case, recomputes
the metrics using the original frozen-model references, selects STR/MOR/BAL
actions for a dense power-loss grid, and independently verifies every selected
action through the same connected evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from centroid_pilot.backend import make_backend
from centroid_pilot.common import (
    atomic_json,
    identity,
    pareto_indices3,
    read_config,
    read_states,
    require_preflight,
    selections,
    sha256,
    write_rows,
)
from centroid_pilot.postprocess import case_dir, convergence_rows, metric_table
from postprocess_cached import load_union_cached


BUDGETS = tuple(round(x / 100.0, 2) for x in range(11))
TAUS = (1.0, 1.025, 1.05)


def _record_row(
    state, mode, budget, tau, tag, ev, action, metrics, pmax, rho_bal, anchors=None
):
    row = {
        "farm": state.farm,
        "centroid": state.centroid,
        "wd": state.wd,
        "strategy": tag,
        "mode": mode,
        "budget": budget,
        "tau": tau,
        "power_kw": float(np.sum(ev["power"])),
        "pbest_kw": pmax,
        "reference": "best_found_feasible_power",
        "status": "VERIFIED",
        **dict(zip(("power_loss", "JS", "JM", "ES", "EM"), metrics.tolist())),
    }
    if anchors is not None:
        ideal, spans = anchors
        q = (metrics[1:3] - ideal) / spans
        row["JB"] = float(q.max() + rho_bal / 2 * q.sum())
    row.update({f"yaw_T{i + 1}": float(action[i]) for i in range(5)})
    row.update({f"pitch_T{i + 1}": float(action[i + 5]) for i in range(5)})
    return row


def extract_case(state, config, backend, root, fingerprint):
    folder = case_dir(root, state)
    out = folder / "dense_budget_0to10"
    out.mkdir(exist_ok=True)

    records, seed_data = load_union_cached(folder, config, fingerprint)
    engine = make_backend(backend, config)
    greedy_ev = engine.evaluate(state, np.zeros(10))
    greedy_loads = greedy_ev["loads"]
    pmax = max(float(np.sum(record["power"])) for record in records)
    metrics = metric_table(records, greedy_loads, pmax)
    imax = int(np.argmax([np.sum(record["power"]) for record in records]))
    greedy_index = next(i for i, record in enumerate(records) if np.array_equal(record["x"], np.zeros(10)))

    dense_config = dict(config)
    dense_config["budgets"] = list(BUDGETS)
    dense_config["taus"] = list(TAUS)

    choices = [("Greedy", None, None, greedy_index, None), ("Pbest", None, None, imax, None)]
    missing = []
    for budget in BUDGETS:
        for tau in TAUS:
            chosen, anchors = selections(metrics, dense_config, budget, tau)
            if not chosen:
                missing.append(
                    {
                        "farm": state.farm,
                        "centroid": state.centroid,
                        "budget": budget,
                        "tau": tau,
                        "status": "NO_FEASIBLE_CANDIDATE_FOUND",
                    }
                )
                continue
            for mode, index in chosen.items():
                choices.append((mode, budget, tau, index, anchors))

    selected = []
    turbines = []
    verified = []
    tolerance = config["feasibility_tolerance"]
    for mode, budget, tau, index, anchors in choices:
        record = records[index]
        action = record["x"]
        ev = engine.evaluate(state, action)
        current = metric_table([{**ev, "x": action}], greedy_loads, pmax)[0]
        if not np.allclose(current, metrics[index], rtol=1e-8, atol=1e-9):
            raise RuntimeError("Selected dense-budget action does not reproduce archive metrics")
        if budget is not None and (
            current[0] > budget + tolerance or max(current[3:5]) > tau + tolerance
        ):
            raise RuntimeError("Selected dense-budget action fails independent constraint check")
        tag = mode if budget is None else f"{mode}{round(100 * budget)}_tau{tau:.3f}"
        row = _record_row(
            state,
            mode,
            budget,
            tau,
            tag,
            ev,
            action,
            current,
            pmax,
            config["rho_bal"],
            anchors,
        )
        selected.append(row)
        verified.append(
            {
                "strategy": tag,
                "action": action.tolist(),
                "budget": budget,
                "tau": tau,
                "metrics": current.tolist(),
            }
        )
        ratio = ev["loads"] / greedy_loads
        for turbine in range(5):
            turbines.append(
                {
                    "farm": state.farm,
                    "centroid": state.centroid,
                    "strategy": tag,
                    "turbine": turbine + 1,
                    "yaw_deg": float(action[turbine]),
                    "pitch_deg": float(action[turbine + 5]),
                    "power_kw": float(ev["power"][turbine]),
                    "ws_eff": float(ev["ws"][turbine]),
                    "ti_eff_pct": float(ev["ti"][turbine]),
                    **{
                        name: float(ev["loads"][turbine, k])
                        for k, name in enumerate(("DEL_TB", "DEL_BR", "DEL_YB", "DEL_Fmax"))
                    },
                    **{
                        name: float(ratio[turbine, k])
                        for k, name in enumerate(("ratio_TB", "ratio_BR", "ratio_YB", "ratio_Fmax"))
                    },
                }
            )

    # Store both three-objective nondominated databases in transparent CSV form.
    front_rows = []
    for formulation, columns in (("mean", (0, 1, 2)), ("envelope", (0, 3, 4))):
        region = np.flatnonzero(
            (metrics[:, 0] >= -tolerance) & (metrics[:, 0] <= max(BUDGETS) + tolerance)
        )
        front = region[pareto_indices3(metrics[region][:, columns])]
        for index in front:
            action = records[index]["x"]
            front_rows.append(
                {
                    "farm": state.farm,
                    "centroid": state.centroid,
                    "formulation": formulation,
                    "power_kw": float(np.sum(records[index]["power"])),
                    **dict(zip(("power_loss", "JS", "JM", "ES", "EM"), metrics[index].tolist())),
                    **{f"yaw_T{i + 1}": float(action[i]) for i in range(5)},
                    **{f"pitch_T{i + 1}": float(action[i + 5]) for i in range(5)},
                }
            )

    write_rows(out / "selected.csv", selected)
    write_rows(out / "selected_turbines.csv", turbines)
    write_rows(out / "missing_selections.csv", missing)
    write_rows(out / "pareto_front_database.csv", front_rows)
    write_rows(out / "convergence.csv", convergence_rows(seed_data, dense_config, greedy_loads, pmax))
    atomic_json(
        out / "selection.json",
        {
            "fingerprint": fingerprint,
            "state": state.__dict__,
            "pbest_kw": pmax,
            "budgets": BUDGETS,
            "taus": TAUS,
            "selection_method": "epsilon constraints applied to the completed three-seed evolutionary archive",
            "polish": False,
            "selected": verified,
            "missing": missing,
            "candidate_count": len(records),
            "front_row_count": len(front_rows),
            "backend": backend,
            "synthetic": backend == "analytic",
            "note": "No feasible candidate found is not proof of infeasibility; Pbest is not a global certificate.",
        },
    )
    print(
        f"Dense database {state.key}: {len(records)} candidates, "
        f"{len(front_rows)} Pareto rows, {len(selected)} verified selections, {len(missing)} empty cells",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract 0--10% dense power/load database")
    parser.add_argument("--config", default="config/pilot.json")
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--out", default="results/centroid_pilot_v2")
    parser.add_argument("--case", required=True, type=int)
    parser.add_argument("--backend", choices=("frozen", "analytic"), default="frozen")
    args = parser.parse_args()

    config = read_config(args.config)
    states = read_states(args.centroids)
    if not 1 <= args.case <= len(states):
        parser.error(f"--case must be 1..{len(states)}")
    ident = identity(config, args.centroids, args.backend)
    require_preflight(args.out, ident)

    provenance = {
        "optimization_fingerprint": ident["fingerprint"],
        "extractor": str(Path(__file__).resolve()),
        "extractor_sha256": sha256(__file__),
        "cached_loader_sha256": sha256("postprocess_cached.py"),
        "budgets": list(BUDGETS),
        "taus": list(TAUS),
        "scope": "Selection and Pareto extraction from completed archives; no model fitting or new optimization",
    }
    marker = Path(args.out) / "dense_budget_0to10.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Dense-budget extractor provenance changed within this output directory")
    atomic_json(marker, provenance)
    extract_case(states[args.case - 1], config, args.backend, args.out, ident["fingerprint"])


if __name__ == "__main__":
    main()
