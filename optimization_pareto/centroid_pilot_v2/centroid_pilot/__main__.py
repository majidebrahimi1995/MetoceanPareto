from __future__ import annotations

import argparse
from pathlib import Path

from .common import identity, read_config, read_states, require_preflight


def main():
    parser = argparse.ArgumentParser(description='Frozen-model centroid-first Pareto pilot')
    parser.add_argument('command',choices=('inspect-models','preflight','run-task','merge','scan','report','task-map'))
    parser.add_argument('--config',default='config/pilot.json')
    parser.add_argument('--centroids',required=True,help='Explicit centroid-source choice; no silent default')
    parser.add_argument('--backend',choices=('frozen','analytic'),default='frozen')
    parser.add_argument('--out',default='results/centroid_pilot_v2')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--task-id',type=int,help='1-based (condition, restart) task, 1..30')
    parser.add_argument('--case',type=int,help='1-based layout/centroid case, 1..10; omit to merge/scan all')
    parser.add_argument('--wd-offset',type=int,default=0,choices=range(-5,6))
    parser.add_argument('--polish',action='store_true')
    parser.add_argument('--mode',choices=('fixed-offset','fixed-heading'),default='fixed-offset')
    parser.add_argument('--yaw-error-sign',type=int,choices=(-1,1))
    args = parser.parse_args()
    c = read_config(args.config)
    states = read_states(args.centroids)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'inspect-models':
        from .preflight import inspect_models
        inspect_models(c)
        return
    if args.command == 'task-map':
        for i,s in enumerate(states):
            for j,seed in enumerate(c['seeds']):
                print(f'{i*len(c["seeds"])+j+1}\t{s.key}\t{seed}\tWD={s.wd}')
        return
    ident = identity(c,args.centroids,args.backend)
    if args.backend == 'analytic':
        print('SYNTHETIC SOFTWARE TEST: not the wind-farm model',flush=True)
    if args.command == 'preflight':
        from .preflight import preflight
        preflight(states,c,args.backend,args.out,ident,args.config)
        return
    require_preflight(args.out,ident)
    if args.command == 'run-task':
        from .search import optimize
        from .postprocess import case_dir
        if args.task_id is None or not 1 <= args.task_id <= len(states)*len(c['seeds']):
            parser.error('--task-id must select one entry in task-map')
        i,j = divmod(args.task_id-1,len(c['seeds']))
        s,seed = states[i],c['seeds'][j]
        folder = case_dir(args.out,s,args.wd_offset)/f'seed_{seed}'
        optimize(s.shifted(args.wd_offset),c,args.backend,args.workers,folder,seed,ident['fingerprint'])
    elif args.command in ('merge','scan'):
        if args.case is not None and not 1 <= args.case <= len(states):
            parser.error('--case must be 1..10')
        chosen = [states[args.case-1]] if args.case else states
        from .postprocess import merge,scan
        for s in chosen:
            if args.command == 'merge':
                merge(s.shifted(args.wd_offset),c,args.backend,args.out,ident['fingerprint'],args.wd_offset,args.polish)
            else:
                if args.wd_offset != 0:
                    parser.error('scan uses the exact centroid; omit --wd-offset')
                scan(s,c,args.backend,args.out,ident['fingerprint'],args.mode,args.yaw_error_sign)
    elif args.command == 'report':
        from .report import report
        report(args.out,states,c,ident['fingerprint'])


if __name__ == '__main__':
    main()
