"""Publication figures drawn only from completed replay evidence."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
DEST=ROOT/'manuscript/figures';DEST.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(DEST/f'{name}.svg',bbox_inches='tight')
    fig.savefig(DEST/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(DEST/f'{name}.png',dpi=900,bbox_inches='tight')
    fig.savefig(DEST/f'{name}_preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
def main():
    data=json.loads((ROOT/'manuscript/article_evidence.json').read_text())
    rows=data['main']
    if len(rows)!=9:raise ValueError(f'Need nine physical cells, got {len(rows)}')
    labels=[f'Panel {r["panel"]+1}  |  '+('28 ToU' if r['battery_count']==28 else ('55 ToU' if r['tariff_mix']=='all_tou' else '55 mixed')) for r in rows]
    fig,ax=plt.subplots(1,3,figsize=(10.2,4.4),sharey=True,gridspec_kw={'wspace':.23})
    specs=[('cost_owner_cents','Operating cost difference\n(AUD cents / battery owner / day)',False),('voltage_pp','Voltage screening difference\n(percentage points)',True),('peak_kw','Mean daily import peak difference\n(kW)',True)]
    for k,(key,label,nested) in enumerate(specs):
        for i,r in enumerate(rows):
            q=r[key]['delta_dfl_minus_mse'] if nested else r[key]
            lo,hi=q['ci95'];m=q['mean'];color=['#1e5e8f','#a35125','#43735f'][r['panel']]
            ax[k].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=color,capsize=2.2,markersize=4.5,lw=1.2)
        ax[k].axvline(0,color='.35',lw=.8,ls='--');ax[k].set_xlabel(label)
        ax[k].set_yticks(range(9),labels);ax[k].grid(axis='x',alpha=.18)
        for sep in [2.5,5.5]:ax[k].axhline(sep,color='.85',lw=.6)
    ax[0].invert_yaxis();fig.subplots_adjust(left=.19,bottom=.18,top=.98)
    save(fig,'figure1_paired_effects')
    mf=ROOT/'results/mechanism_physical_vs_msecont'
    z={k:np.load(mf/f'{k}.npz') for k in ['mse','common_only','idiosyncratic_only','dfl']}
    clean=z['mse']['clean_day'];hours=np.arange(48)/2+.25
    fig,ax=plt.subplots(1,2,figsize=(10.2,3.3))
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz');ids=json.loads((ROOT/'datasets/splits_v1.json').read_text())['primary_panels'][0]
    hh=[list(a['household_id']).index(h) for h in ids];dd=np.flatnonzero(a['split_code']==2)
    truth=a['truth_pv_kw'][hh][:,dd][:,clean].mean((0,1))
    ax[0].plot(hours,truth,label='Metered PV',color='black',ls=':',lw=1.7)
    for m,label,col in [('msecont','Continued MSE','#1e5e8f'),('dfl','Decision focused','#a35125')]:
        s=np.load(ROOT/f'results/dispatch/{m}_tou_seed11_test_physical.npz');ii=[list(s['household_id']).index(h) for h in ids]
        ax[0].plot(hours,s['forecast_pv_kw'][ii][:,clean].mean((0,1)),label=label,color=col,lw=1.6)
    for m,label,col,ls in [('mse','Continued MSE','#1e5e8f','-'),('common_only','Common correction','#43735f','--'),('dfl','Decision focused','#a35125','-')]:
        ax[1].plot(hours,z[m]['aggregate_battery_kw'][clean].mean(0),label=label,color=col,ls=ls,lw=1.6)
    ax[0].set_ylabel('Mean PV signal (kW per household)');ax[1].set_ylabel('Aggregate battery power (kW)')
    for aa in ax:
        aa.set_xlabel('Source-clock hour');aa.set_xticks([0,6,12,18,24]);aa.set_xlim(0,24);aa.grid(alpha=.18);aa.legend(frameon=False,fontsize=8)
    ax[1].axhline(0,color='.35',lw=.7);fig.tight_layout()
    save(fig,'figure2_signal_and_actions')
    print(json.dumps({'figures':2,'raster_dpi':900,'vector_formats':['svg','pdf'],'days':int(clean.sum())}))
if __name__=='__main__':main()
