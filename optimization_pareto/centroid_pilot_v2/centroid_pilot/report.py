from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atomic_json
from .postprocess import case_dir


def report(root, states, config, fingerprint):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(root)
    out = root/'report'
    out.mkdir(exist_ok=True)
    selected,conv,scans = [],[],[]
    present,missing = [],[]
    fig,axes = plt.subplots(2,5,figsize=(15,6),sharex=True,sharey=True)
    synthetic = False
    for i,state in enumerate(states):
        folder = case_dir(root,state)
        meta = folder/'merged/selection.json'
        ax = axes.flat[i]
        ax.set_title(state.key,fontsize=10)
        ax.set_xlabel('Structural DEL ratio, JS')
        ax.set_ylabel('Mooring DEL ratio, JM')
        if not meta.exists():
            missing.append(state.key)
            ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center')
            continue
        metadata = json.loads(meta.read_text())
        if metadata['fingerprint'] != fingerprint:
            raise RuntimeError('Mismatched model/config version in '+str(meta))
        present.append(state.key)
        synthetic |= metadata['synthetic']
        z = np.load(folder/'merged/archive.npz',allow_pickle=False)
        m = z['metrics'][z['nondominated_indices']]
        m = m[m[:,0] <= max(config['budgets'])]
        ax.scatter(m[:,1],m[:,2],c=100*m[:,0],s=4,cmap='viridis',vmin=0,vmax=6)
        s = pd.read_csv(folder/'merged/selected.csv')
        selected.append(s)
        relevant = s[s['tau'].sub(1.025).abs()<1e-9]
        for mode,marker in (('STR','^'),('MOR','s'),('BAL','o')):
            sub = relevant[relevant['mode']==mode]
            ax.scatter(sub.JS,sub.JM,marker=marker,facecolors='none',edgecolors='black',s=35)
        ax.grid(alpha=.2)
        cv = pd.read_csv(folder/'merged/convergence.csv')
        cv.insert(0,'case',state.key)
        conv.append(cv)
        scan = folder/'scan_fixed-offset.csv'
        if scan.exists():
            scans.append(pd.read_csv(scan))
    label = 'SYNTHETIC SOFTWARE TEST - NOT SCIENTIFIC RESULTS' if synthetic else 'Centroid Pareto diagnostics; colour = power loss, 0-6%'
    fig.suptitle(label,fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'centroid_pareto.png',dpi=180)
    plt.close(fig)
    if selected:
        pd.concat(selected,ignore_index=True).to_csv(out/'all_selected.csv',index=False)
        pd.concat(conv,ignore_index=True).to_csv(out/'all_convergence.csv',index=False)
    if scans:
        frame = pd.concat(scans,ignore_index=True)
        frame.to_csv(out/'all_scans.csv',index=False)
        fig,axes = plt.subplots(2,5,figsize=(15,6),sharex=True)
        for i,state in enumerate(states):
            ax = axes.flat[i]
            ax.set_title(state.key,fontsize=10)
            sub = frame[(frame.farm==state.farm)&(frame.centroid==state.centroid)]
            for name,g in sub[sub.status=='VALID'].groupby('strategy'):
                ax.plot(g.delta_wd,np.maximum(g.ES,g.EM),label=name,linewidth=1)
            ax.axhline(1.025,color='black',ls='--',lw=.8)
            ax.set_xlabel('Direction offset [deg]')
            ax.set_ylabel('Largest local DEL ratio')
            ax.grid(alpha=.2)
        handles,labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
        fig.suptitle(label)
        fig.tight_layout(rect=(0,.12,1,.97))
        fig.savefig(out/'direction_envelopes.png',dpi=180)
        plt.close(fig)
    atomic_json(out/'completeness.json',{'present':present,'missing':missing,
        'synthetic':synthetic,'fingerprint':fingerprint})
    (out/'RESULTS_TO_WRITE.md').write_text(
        '# Results status\n\n'+('SYNTHETIC TEST ONLY.\n\n' if synthetic else '')+
        f'Completed centroid cases: {len(present)}/10. Pending: {", ".join(missing) or "none"}.\n\n'+
        'Use all_convergence.csv to compare horizons and independent restarts before interpreting the selected values. '
        'Empty selections are search outcomes, not proof that the constraints are infeasible. '
        'No case is automatically excluded because its control benefit is small. '
        'The proposed results narrative is in docs/results_plan.md.\n')
    print(f'Report prepared for {len(present)}/10 cases: {out}',flush=True)
