from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


class InvalidState(ValueError):
    """Expected physical/domain failure; programming errors must propagate."""


@dataclass(frozen=True)
class State:
    farm: str
    centroid: str
    ws: float
    ti: float
    wd: float
    hs: float
    tp: float

    @property
    def key(self):
        return f"{self.farm}_{self.centroid}"

    def shifted(self, delta):
        return replace(self, wd=(self.wd + delta) % 360.0)


def read_states(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 5 or {r['centroid'] for r in rows} != {f'C{i}' for i in range(1, 6)}:
        raise ValueError('Centroid file must contain C1..C5 exactly once')
    states = [State(farm, r['centroid'], **{k: float(r[k]) for k in ('ws','ti','wd','hs','tp')})
              for farm in ('Rectangle','Tampen') for r in rows]
    for s in states:
        if not np.isfinite([s.ws,s.ti,s.wd,s.hs,s.tp]).all() or not 0 <= s.wd < 360:
            raise ValueError(f'Invalid centroid {s}')
    return states


def read_config(path):
    c = json.loads(Path(path).read_text())
    platform_root = Path(__file__).resolve().parents[3]
    for key in ('main','bundle','base_model','sea_model','runner','pywake_runner'):
        c[key] = str((platform_root / c[key]).resolve())
    if c['population'] < 8 or c['generations'] < 2 or not c['seeds']:
        raise ValueError('Invalid search size')
    if len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Restart seeds must be distinct')
    if any(g < 1 or g > c['generations'] for g in c['checkpoints']):
        raise ValueError('Checkpoints must lie within the configured generation budget')
    if c['generations'] not in c['checkpoints']:
        raise ValueError('Include the final generation in checkpoints')
    return c


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('status\nNO_ROWS\n')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', newline='') as stream:
        w = csv.DictWriter(stream, fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


@contextmanager
def lock(path):
    """POSIX advisory lock released on process exit, including killed jobs."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process owns {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def identity(c, centroid_file, backend):
    """Includes code, resolved runtime, models and data; detects CURRENT changes."""
    files = {str(p.resolve()): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    files[str(Path(centroid_file).resolve())] = sha256(centroid_file)
    if backend == 'frozen':
        from .backend import FrozenBackend
        engine = FrozenBackend(c)
        paths = [Path(p) for p in engine.evaluator.provenance()['asset_files']]
        paths += [Path(__file__).resolve().parents[1] / 'postprocess_cached.py',
                  Path(__file__).resolve().parents[1] / 'dense_budget_extract.py']
        for p in paths:
            files[str(p)] = sha256(p)
    packages = {}
    for name in ('numpy','scipy','pandas','joblib','scikit-learn','py-wake','pymoo'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unavailable'
    payload = {'config':c,'files':files,'backend':backend,
               'python':platform.python_version(),'packages':packages}
    return {'fingerprint':digest_json(payload), **payload}


def require_preflight(root, ident):
    path = Path(root) / 'preflight.json'
    if not path.exists():
        raise RuntimeError('Run preflight into this output directory first')
    old = json.loads(path.read_text())
    if old['fingerprint'] != ident['fingerprint'] or old['status'] != 'PASS':
        raise RuntimeError('Preflight failed or code/config/model/environment changed; use a new output directory')


def controls(c):
    return np.array([c['yaw_bounds']]*5 + [c['pitch_bounds']]*5, float)


def metrics(power, loads, greedy_loads, pmax):
    ratios = np.asarray(loads) / np.asarray(greedy_loads)
    return np.array([1-float(np.sum(power))/pmax, ratios[:,:3].mean(),
                     ratios[:,3].mean(), ratios[:,:3].max(), ratios[:,3].max()])


def selections(m, c, budget, tau):
    eps = c['feasibility_tolerance']
    mask = (m[:,0] <= budget+eps) & (m[:,3] <= tau+eps) & (m[:,4] <= tau+eps)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {}, None
    # Deterministic tie breaking: primary objective, other load, power loss.
    si = idx[np.lexsort((m[idx,0],m[idx,2],m[idx,1]))[0]]
    mi = idx[np.lexsort((m[idx,0],m[idx,1],m[idx,2]))[0]]
    ideal = np.array([m[si,1], m[mi,2]])
    spans = np.maximum([m[mi,1]-m[si,1],m[si,2]-m[mi,2]],c['payoff_floor'])
    q = (m[idx,1:3]-ideal)/spans
    score = q.max(axis=1) + c['rho_bal']/2*q.sum(axis=1)
    bi = idx[np.lexsort((m[idx,0], score))[0]]
    return {'STR':int(si),'MOR':int(mi),'BAL':int(bi)}, (ideal,spans)


def action_distance(a, b, c):
    widths = np.diff(controls(c), axis=1).ravel()
    return float(np.sqrt(np.mean(((np.asarray(a)-b)/widths)**2)))


def pareto_indices3(values):
    """Exact 3-objective minimization skyline, O(n log n), O(n) storage.

    Drop repeated objective triples deterministically. Avoids an n-by-n dominance
    matrix when a complete evaluated archive contains tens of thousands of points.
    """
    f = np.asarray(values,float)
    if f.ndim != 2 or f.shape[1] != 3 or not np.isfinite(f).all():
        raise ValueError('Expected finite (n,3) objective array')
    if not len(f):
        return np.array([],dtype=int)
    order = np.lexsort((f[:,2],f[:,1],f[:,0]))
    ranks = np.searchsorted(np.unique(f[:,1]),f[:,1])+1
    tree = np.full(int(ranks.max())+1,np.inf)
    keep = []
    for i in order:
        rank = int(ranks[i])
        minimum = np.inf
        k = rank
        while k:
            minimum = min(minimum,tree[k])
            k -= k & -k
        if minimum > f[i,2]:
            keep.append(i)
        k = rank
        while k < len(tree):
            tree[k] = min(tree[k],f[i,2])
            k += k & -k
    return np.asarray(keep,int)
