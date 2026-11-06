#!/usr/bin/env python3
"""Memory-safe launcher for completed centroid-pilot archives.

This launcher does not alter the optimization, metrics, selections, model, or
stored evaluations.  It replaces only ``load_union`` at runtime so every array
inside each compressed NPZ restart archive is decompressed once.  The original
implementation repeatedly indexed the NpzFile inside a row loop, creating
extreme virtual-memory growth on SGE during postprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from centroid_pilot.common import atomic_json, identity, read_config, read_states, require_preflight, sha256
import centroid_pilot.postprocess as postprocess


def load_union_cached(folder, config, fingerprint):
    """Behavior-preserving replacement for postprocess.load_union.

    Arrays are materialized once per restart.  Row selection, exact action-byte
    deduplication, and the cross-restart determinism check match the canonical
    implementation.
    """
    records = {}
    seed_data = []
    for seed in config["seeds"]:
        directory = Path(folder) / f"seed_{seed}"
        done_path = directory / "done.json"
        if not done_path.exists():
            raise RuntimeError(f"Missing completed restart: {directory}")
        if json.loads(done_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError(f"Fingerprint mismatch: {directory}")

        with np.load(directory / "evaluations.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        seed_data.append((seed, arrays))

        valid_indices = np.flatnonzero(arrays["ok"])
        for i in valid_indices:
            x = arrays["X"][i]
            key = x.tobytes()
            row = {name: arrays[name][i] for name in ("power", "loads", "ws", "ti")}
            if key in records:
                if any(
                    not np.allclose(records[key][name], row[name], rtol=1e-9, atol=1e-9)
                    for name in row
                ):
                    raise RuntimeError("Different predictions for identical action across restarts")
                records[key]["seeds"].add(seed)
            else:
                records[key] = {"x": x, **row, "seeds": {seed}}
    return list(records.values()), seed_data


def main():
    parser = argparse.ArgumentParser(description="Memory-safe centroid postprocessing")
    parser.add_argument("--config", default="config/pilot.json")
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--out", default="results/centroid_pilot_v2")
    parser.add_argument("--case", required=True, type=int, help="1-based layout/centroid case")
    parser.add_argument("--backend", choices=("frozen", "analytic"), default="frozen")
    parser.add_argument("--polish", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)
    states = read_states(args.centroids)
    if not 1 <= args.case <= len(states):
        parser.error(f"--case must be 1..{len(states)}")

    ident = identity(config, args.centroids, args.backend)
    require_preflight(args.out, ident)
    postprocess.load_union = load_union_cached

    provenance_path = Path(args.out) / "postprocess_cached.json"
    provenance = {
        "optimization_fingerprint": ident["fingerprint"],
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256(__file__),
        "scope": "NPZ arrays materialized once; model, evaluations, metrics, constraints and selection logic unchanged",
    }
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if old != provenance:
            raise RuntimeError("Postprocessing launcher provenance changed within this output directory")
    else:
        atomic_json(provenance_path, provenance)

    state = states[args.case - 1]
    postprocess.merge(
        state,
        config,
        args.backend,
        args.out,
        ident["fingerprint"],
        offset=0,
        polish=args.polish,
    )


if __name__ == "__main__":
    main()
