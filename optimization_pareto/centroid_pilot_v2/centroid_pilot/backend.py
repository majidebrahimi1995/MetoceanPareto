from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from .common import InvalidState, State


def target_order(targets):
    """Only four unambiguous channels; never substitute an individual fairlead."""
    result = []
    for words in (('towerbase',),('bladeroot',),('yawbearing',),('fairleadmax','fairtenmax','fairmax')):
        matches = [i for i, t in enumerate(targets)
                   if any(w in ''.join(ch for ch in t.lower() if ch.isalnum()) for w in words)]
        if len(matches) != 1:
            raise RuntimeError(f'Cannot unambiguously identify channel {words} in {targets}')
        result.append(matches[0])
    if len(targets) != 4 or len(set(result)) != 4:
        raise RuntimeError(f'Expected exactly four DEL outputs, got {targets}')
    return result


class AnalyticBackend:
    """SYNTHETIC test fixture. It is not a wind/wave model."""
    def __init__(self, config):
        self.config = config

    def evaluate(self, state, action):
        a = np.asarray(action, float)
        if a.shape != (10,) or not np.isfinite(a).all():
            raise InvalidState('Invalid action')
        y, p = a[:5], a[5:]
        if (np.abs(y)>20+1e-9).any() or (p<0).any() or (p>3).any():
            raise InvalidState('Actuator domain')
        phase = np.deg2rad(state.wd + np.arange(5)*17)
        desired = 7*np.sin(phase)
        power = state.ws**3 * (1 - .00022*(y-desired)**2 - .025*p)
        ws = state.ws*(.85+.1*np.cos(phase))
        ti = state.ti + 2*np.abs(np.sin(phase))
        loads = np.column_stack([
            100*(1+.002*y+.0002*y*y-.04*p)*(1+.1*state.hs),
            90*(1-.001*y+.00018*y*y-.05*p),
            70*(1+.003*y+.0003*y*y-.025*p),
            200*(1-.002*y+.00015*y*y-.035*p)*(1+.2*state.hs)])
        return {'power':power,'loads':loads,'ws':ws,'ti':ti,
                'x':np.arange(5)*1000.,'y':np.zeros(5)}


class FrozenBackend:
    """Thin adapter: all model calculations belong to the canonical runner."""

    def __init__(self, config):
        import importlib.util
        from .common import sha256
        self.config = config
        path = Path(config['runner']).resolve()
        spec = importlib.util.spec_from_file_location('_authoritative_connected_runner',path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if not hasattr(module,'ConnectedEvaluator'):
            raise RuntimeError('The connected evaluator has no ConnectedEvaluator API')
        self.module = module
        self.evaluator = module.ConnectedEvaluator(
            base_model=config['base_model'],sea_model=config['sea_model'],bundle=config['bundle'],
            pywake_runner=config['pywake_runner'],allow_model_change=False)
        self.order = target_order(self.evaluator.targets)

    def evaluate(self, state: State, action):
        a = np.asarray(action,float)
        if a.shape != (10,) or not np.isfinite(a).all():
            raise InvalidState('Expected ten finite control values')
        try:
            value = self.evaluator.evaluate(
                farm=state.farm,ws=state.ws,wd=state.wd,ti=state.ti,hs=state.hs,tp=state.tp,
                yaw=a[:5],pitch=a[5:])
        except self.module.ModelDomainError as exc:
            raise InvalidState(str(exc)) from exc
        frame,aero = value['frame'],value['pywake']
        if frame.Turbine.tolist() != [1,2,3,4,5] or aero.Turbine.tolist() != [1,2,3,4,5]:
            raise RuntimeError('Canonical turbine identity/order must be 1..5')
        columns = ['sea_'+self.evaluator.targets[j] for j in self.order]
        return {'power':aero.Power_kW.to_numpy(float),'loads':frame[columns].to_numpy(float),
                'ws':frame.local_ws_mps.to_numpy(float),'ti':frame.local_ti_pct.to_numpy(float)}


def make_backend(name, config):
    return {'analytic':AnalyticBackend,'frozen':FrozenBackend}[name](config)


_WORKER = None


def init_worker(backend, config):
    global _WORKER
    _WORKER = make_backend(backend,config)


def evaluate_worker(payload):
    state,action = payload
    try:
        return {'ok':True, **_WORKER.evaluate(state,action)}
    except InvalidState as exc:
        return {'ok':False,'reason':str(exc)}
