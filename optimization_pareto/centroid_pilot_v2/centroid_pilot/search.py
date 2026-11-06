from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc

from .backend import evaluate_worker, init_worker
from .common import atomic_json, atomic_npz, controls, lock, metrics


class Bank:
    """Exact-action memoization + complete candidate archive, including invalids.

    Sightings are tracked independently for both objective formulations so that
    horizon comparisons never borrow later-generation points from another search.
    """
    def __init__(self, state, config, backend, workers=1, path=None):
        self.state,self.config,self.path = state,config,Path(path) if path else None
        self.records = []
        self.index = {}
        self.phase,self.generation = 'pmax',0
        self.pool = None
        self.seconds = 0.
        if self.path and self.path.exists():
            d = np.load(self.path,allow_pickle=False)
            for i,x in enumerate(d['X']):
                rec = {k:d[k][i].copy() for k in ('power','loads','ws','ti','seen')}
                rec.update(x=x.copy(),ok=bool(d['ok'][i]),reason=str(d['reason'][i]))
                self.index[x.tobytes()] = len(self.records)
                self.records.append(rec)
        if workers > 1:
            self.pool = ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                initializer=init_worker,initargs=(backend,config))
        else:
            init_worker(backend,config)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True,cancel_futures=True)

    def batch(self, actions):
        actions = np.atleast_2d(np.asarray(actions,dtype=np.float64))
        missing = {}
        for x in actions:
            key = x.tobytes()
            if key not in self.index:
                missing[key] = x.copy()
        started = time.perf_counter()
        jobs = [(self.state,x) for x in missing.values()]
        values = self.pool.map(evaluate_worker,jobs) if self.pool else map(evaluate_worker,jobs)
        for (key,x),v in zip(missing.items(),values):
            rec = {'x':x,'ok':v['ok'],'reason':v.get('reason',''),
                   'seen':np.full(3,-1,dtype=int)}
            for field,shape in (('power',(5,)),('loads',(5,4)),('ws',(5,)),('ti',(5,))):
                rec[field] = np.asarray(v[field],float) if v['ok'] else np.full(shape,np.nan)
            self.index[key] = len(self.records)
            self.records.append(rec)
        self.seconds += time.perf_counter()-started
        result = []
        col = {'pmax':0,'mean':1,'envelope':2,'polish':0}[self.phase]
        for x in actions:
            rec = self.records[self.index[x.tobytes()]]
            seen = rec['seen'][col]
            rec['seen'][col] = self.generation if seen < 0 else min(seen,self.generation)
            result.append(rec)
        return result

    def one(self, x):
        return self.batch(x)[0]

    def save(self):
        if not self.path or not self.records:
            return
        arrays = {k:np.asarray([r[k] for r in self.records])
                  for k in ('power','loads','ws','ti','seen','ok','reason')}
        atomic_npz(self.path,X=np.asarray([r['x'] for r in self.records]),**arrays)


def optimize(state, config, backend, workers, out, seed, fingerprint):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'task.lock'):
        done = out/'done.json'
        if done.exists():
            old = json.loads(done.read_text())
            if old['fingerprint'] != fingerprint:
                raise RuntimeError('Existing task belongs to another configuration')
            print(f'SKIP completed: {out}',flush=True)
            return
        meta = out/'identity.json'
        if meta.exists() and json.loads(meta.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('Refusing to resume an incompatible task')
        atomic_json(meta,{'fingerprint':fingerprint,'seed':seed,'state':state.__dict__})
        bank = Bank(state,config,backend,workers,out/'evaluations.npz')
        start = time.perf_counter()
        bounds = controls(config)
        try:
            greedy = bank.one(np.zeros(10))
            if not greedy['ok']:
                raise RuntimeError('Greedy reference invalid: '+greedy['reason'])
            greedy_loads = greedy['loads']
            def loss(x):
                r = bank.one(x)
                return -float(r['power'].sum()) if r['ok'] else 1e12
            def parallel_map(_func, iterable):
                return [-float(r['power'].sum()) if r['ok'] else 1e12 for r in bank.batch(list(iterable))]
            # One independent DE run per restart; merge pools all three searches.
            de = differential_evolution(loss,bounds.tolist(),seed=seed,
                maxiter=config['de_maxiter'],popsize=config['de_popsize'],
                x0=np.zeros(10),workers=parallel_map,updating='deferred',
                polish=False,tol=1e-7,atol=1e-6)
            valid = [r for r in bank.records if r['ok']]
            p_rec = max(valid,key=lambda r:r['power'].sum())
            pmax = float(p_rec['power'].sum())
            if pmax <= 0:
                raise RuntimeError('No positive feasible power reference')
            bank.save()
            atomic_json(out/'power_search.json',{'power_kw':pmax,'action':p_rec['x'].tolist(),
                'de_success':bool(de.success),'message':str(de.message),'nfev':int(de.nfev),
                'reference_type':'best-found feasible coupled-domain power; not a certified global maximum'})

            class FarmProblem(Problem):
                def __init__(self, formulation):
                    self.formulation = formulation
                    super().__init__(n_var=10,n_obj=3,n_ieq_constr=1,xl=bounds[:,0],xu=bounds[:,1])
                def _evaluate(self,X,out,*args,**kwargs):
                    bank.phase = self.formulation
                    rows = bank.batch(X)
                    F = np.full((len(rows),3),1e6)
                    G = np.ones((len(rows),1))
                    for i,r in enumerate(rows):
                        if r['ok']:
                            m = metrics(r['power'],r['loads'],greedy_loads,pmax)
                            F[i] = m[[0,1,2] if self.formulation == 'mean' else [0,3,4]]
                            G[i,0] = -1.
                    if bank.generation >= 10 and np.mean(G[:,0]>0)>config['invalid_fraction_abort']:
                        bank.save()
                        raise RuntimeError('Most candidate states are outside model support; inspect evaluations.npz')
                    out['F'],out['G'] = F,G

            class Save(Callback):
                def __init__(self, formulation):
                    super().__init__()
                    self.formulation = formulation
                def notify(self, algorithm):
                    gen = int(algorithm.n_gen)
                    if gen % 10 == 0 or gen in config['checkpoints']:
                        bank.save()
                        atomic_json(out/'progress.json',{'formulation':self.formulation,'generation':gen,
                            'unique_evaluations':len(bank.records),'elapsed_seconds':time.perf_counter()-start})
                        print(f'{state.key} seed={seed} {self.formulation} generation={gen} unique={len(bank.records)}',flush=True)
                    if gen in config['checkpoints']:
                        atomic_npz(out/f'{self.formulation}_g{gen:04d}_population.npz',
                            X=algorithm.pop.get('X'),F=algorithm.pop.get('F'),G=algorithm.pop.get('G'))
                    bank.generation = gen+1

            for j,form in enumerate(('mean','envelope')):
                bank.phase,bank.generation = form,1
                sampler = qmc.LatinHypercube(d=10,seed=seed+100*j)
                initial = qmc.scale(sampler.random(config['population']),bounds[:,0],bounds[:,1])
                initial[0],initial[1] = np.zeros(10),p_rec['x']
                algorithm = NSGA2(pop_size=config['population'],sampling=initial,
                    crossover=SBX(prob=.9,eta=15),mutation=PM(eta=20),eliminate_duplicates=True)
                minimize(FarmProblem(form),algorithm,('n_gen',config['generations']),seed=seed+100*j,
                    callback=Save(form),verbose=False,copy_algorithm=False,copy_termination=False)
                bank.save()
            atomic_json(done,{'fingerprint':fingerprint,'state':state.__dict__,'seed':seed,
                'seconds':time.perf_counter()-start,'evaluation_seconds':bank.seconds,
                'unique_evaluations':len(bank.records),
                'invalid_evaluations':sum(not r['ok'] for r in bank.records)})
        finally:
            bank.save()
            bank.close()
