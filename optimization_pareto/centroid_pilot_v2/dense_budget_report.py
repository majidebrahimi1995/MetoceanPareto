#!/usr/bin/env python3
"""Combine the ten checked dense-budget case outputs without re-evaluating models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from centroid_pilot.common import atomic_json, identity, read_config, read_states, require_preflight
from centroid_pilot.postprocess import case_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/pilot.json')
    ap.add_argument('--centroids', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backend', choices=('frozen', 'analytic'), default='frozen')
    args = ap.parse_args()
    config = read_config(args.config)
    states = read_states(args.centroids)
    ident = identity(config, args.centroids, args.backend)
    require_preflight(args.out, ident)
    destination = Path(args.out) / 'dense_budget_0to10'
    destination.mkdir(parents=True, exist_ok=True)
    tables = ('selected', 'selected_turbines', 'missing_selections',
              'pareto_front_database', 'convergence')
    for name in tables:
        parts = []
        for state in states:
            case = case_dir(args.out, state) / 'dense_budget_0to10'
            meta = case / 'selection.json'
            if not meta.is_file():
                raise FileNotFoundError(f'Dense case is missing: {state.key}: {meta}')
            if json.loads(meta.read_text())['fingerprint'] != ident['fingerprint']:
                raise RuntimeError(f'Dense case fingerprint mismatch: {state.key}')
            file = case / (name + '.csv')
            if not file.is_file():
                raise FileNotFoundError(file)
            frame = pd.read_csv(file)
            if list(frame.columns) == ['status'] and frame['status'].tolist() == ['NO_ROWS']:
                continue
            parts.append(frame)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(destination / (('all_' if name == 'selected' else '') + name + '.csv'), index=False)
        elif name == 'missing_selections':
            (destination / (name + '.csv')).write_text('farm,centroid,budget,tau,status\n')
        else:
            raise RuntimeError(f'No rows found for {name}')
    selected = pd.read_csv(destination / 'all_selected.csv')
    missing = pd.read_csv(destination / 'missing_selections.csv')
    atomic_json(destination / 'summary.json', {
        'fingerprint': ident['fingerprint'],
        'backend': args.backend,
        'synthetic': args.backend == 'analytic',
        'cases': [state.key for state in states],
        'selected_rows': int(len(selected)),
        'unique_actions': int(selected.drop_duplicates(['farm','centroid'] +
                         [f'yaw_T{i}' for i in range(1,6)] +
                         [f'pitch_T{i}' for i in range(1,6)]).shape[0]),
        'missing_cells': int(len(missing)),
    })
    print(f'Complete dense database: {destination}')


if __name__ == '__main__':
    main()
