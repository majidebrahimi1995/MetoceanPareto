#!/usr/bin/env python3
"""Portable entry point for the frozen wake, load, optimization, and validation tools."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PILOT = ROOT / 'optimization_pareto/centroid_pilot_v2'
DEFAULT_CENTROIDS = PILOT / 'config/centroids_validation_exact_wd167p830.csv'
DEFAULT_CONFIG = PILOT / 'config/pilot.json'
MODELS = ROOT / 'models/october_20261013'
CHECKS = {
    'models/october_20261013/base.joblib': '5260b2f50857f53f60a83faae56a210a251e3f51d9b684ab2456421c341ae679',
    'models/october_20261013/sea.joblib': '2f57f175b33358c58fa03eb480b3b4165a66a5c2d8ba174971c00ef8f55c9676',
    'surrogate/smooth_del_v1/surrogate_core.py': '97781ba6442a3b2ef696d85c5f74aa8039c881ce8a730195ef89177cccfe3c32',
    'deployment/CURRENT/pywake/openfast_absolute_3d_pywake_table_TI6_PROJECTED_CT_BOUNDED.csv': 'bf813b079c40d0aed25a6fc423443833452c9de8536a6864df594adeef6e7782',
    'deployment/CURRENT/pywake/pitch_ct_tilt_map.csv': '319d4fb889252ed65a25339affe84ded6fe9d20c25d41fb5ba6f6c33d52df4c8',
    'deployment/CURRENT/pywake/pywake_absolute3d_calibration.json': '47eacccd16decfe53f4d8991fc2a70703548246fab6303796895e3aff379977e',
    'deployment/CURRENT/runtime/calibrate_pywake_t2.py': '1c9639ccb77ad2fd548ce96562caa41ceb24113b28356f523bc39317c94fff88',
    'deployment/CURRENT/runtime/openfast_absolute_3d_turbine.py': '2630d8cec0588dff84c5923aeac178bdeb837a038c7c7a9b0bf7e2e27a5c46f9',
    'deployment/CURRENT/runtime/rotation_convention.py': 'bcf5b1d8b6a47a0fa7f5569c2a2a0d599dc96e9f8d61a1e2fcc236b9d42a701c',
}
VERSIONS = {'numpy': '1.26.4', 'scipy': '1.11.4', 'pandas': '2.2.3',
            'scikit-learn': '1.4.2', 'joblib': '1.5.3', 'py-wake': '2.6.20',
            'pymoo': '0.6.1.5', 'matplotlib': '3.10.9'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def doctor():
    print('MetoceanPareto root:', ROOT)
    print('Python:', sys.executable, sys.version.split()[0])
    failures = []
    if sys.version_info[:2] != (3, 11):
        failures.append('Python 3.11 is required for the recorded environment')
    for name, expected in VERSIONS.items():
        try:
            found = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found = 'MISSING'
        print(f'{name}: {found} (recorded {expected})')
        if found != expected:
            failures.append(f'{name}: expected {expected}, found {found}')
    for relative, expected in CHECKS.items():
        path = ROOT / relative
        found = sha256(path) if path.is_file() else 'MISSING'
        print(f'{relative}: {"OK" if found == expected else "MISMATCH"}')
        if found != expected:
            failures.append(f'{relative}: expected {expected}, found {found}')
    if failures:
        raise SystemExit('Environment/assets need attention:\n  ' + '\n  '.join(failures))
    print('PASS: recorded environment and frozen data hashes. Run the parity example next.')


def run(command, cwd=ROOT):
    print('$', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)


def pilot_command(name, args, *, task_id=None, case=None):
    cmd = [sys.executable, '-m', 'centroid_pilot', name,
           '--config', args.config, '--centroids', args.centroids,
           '--out', args.output, '--backend', args.backend]
    if task_id is not None:
        cmd.extend(['--task-id', task_id, '--workers', args.workers])
    if case is not None:
        cmd.extend(['--case', case])
    return cmd


def pareto(args):
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    args.output = str(Path(args.output).resolve())
    args.config = str(Path(args.config).resolve())
    args.centroids = str(Path(args.centroids).resolve())
    if args.backend == 'frozen':
        doctor()
    common = ['--config', args.config, '--centroids', args.centroids,
              '--out', args.output, '--backend', args.backend]
    stages = ['preflight', 'search', 'merge', 'finalize', 'dense', 'dense-report']
    requested = stages if args.stage == 'all' else [args.stage]
    if 'preflight' in requested:
        run(pilot_command('preflight', args), PILOT)
    if 'search' in requested:
        ids = [args.task_id] if args.task_id else range(1, 31)
        for task in ids:
            run(pilot_command('run-task', args, task_id=task), PILOT)
    if 'merge' in requested:
        cases = [args.case] if args.case else range(1, 11)
        for case in cases:
            run([sys.executable, 'postprocess_cached.py', *common, '--case', case, '--polish'], PILOT)
    if 'finalize' in requested:
        run(pilot_command('scan', args), PILOT)
        run(pilot_command('report', args), PILOT)
    if 'dense' in requested:
        cases = [args.case] if args.case else range(1, 11)
        for case in cases:
            run([sys.executable, 'dense_budget_extract.py', *common, '--case', case], PILOT)
    if 'dense-report' in requested:
        run([sys.executable, 'dense_budget_report.py', *common], PILOT)


def main():
    # Forward the case runner's own options unchanged. argparse REMAINDER does
    # not accept an unknown --farm option at the start of a subcommand.
    if len(sys.argv) > 1 and sys.argv[1] in ('wake', 'loads'):
        relative = ('pywake/bin/run_frozen_pywake_case.py' if sys.argv[1] == 'wake'
                    else 'surrogate/smooth_del_v1/run_pywake_sea_surrogate.py')
        options = sys.argv[3:] if sys.argv[2:3] == ['--'] else sys.argv[2:]
        run([sys.executable, ROOT / relative, *options], Path.cwd())
        return
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='check recorded package versions and frozen assets')
    for name in ('wake', 'loads'):
        p = sub.add_parser(name, help='forward arguments to the frozen case runner')
        p.add_argument('arguments', nargs=argparse.REMAINDER)
    p = sub.add_parser('pareto', help='run the exact-centroid optimization pipeline')
    p.add_argument('--stage', choices=('all','preflight','search','merge','finalize','dense','dense-report'), default='all')
    p.add_argument('--centroids', type=Path, default=DEFAULT_CENTROIDS)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1, help='per-search CPU processes')
    p.add_argument('--task-id', type=int, choices=range(1,31))
    p.add_argument('--case', type=int, choices=range(1,11))
    p.add_argument('--backend', choices=('frozen','analytic'), default='frozen')
    p = sub.add_parser('validate', help='compare five-turbine FAST.Farm data with both model branches')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--setpoints', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--environment', type=Path, default=ROOT / 'config/validation_environment.json')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    if args.command == 'doctor':
        doctor()
    elif args.command == 'pareto':
        pareto(args)
    elif args.command in ('wake', 'loads'):
        path = ('pywake/bin/run_frozen_pywake_case.py' if args.command == 'wake'
                else 'surrogate/smooth_del_v1/run_pywake_sea_surrogate.py')
        remainder = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        run([sys.executable, ROOT / path, *remainder], Path.cwd())
    elif args.command == 'validate':
        cmd = [sys.executable, ROOT / 'optimization_pareto/simple_cross_validation/run_all_validation.py',
               '--input', args.input.resolve(), '--setpoints', args.setpoints.resolve(),
               '--output-dir', args.output_dir.resolve(), '--environment', args.environment.resolve(),
               '--workers', args.workers]
        if args.audit_only:
            cmd.append('--audit-only')
        run(cmd)


if __name__ == '__main__':
    main()
