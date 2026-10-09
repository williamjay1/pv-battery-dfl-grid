"""Pair complete topology/mapping identities and summarize prespecified controls."""
from pathlib import Path
import argparse,json
import numpy as np
from analyze_network import metrics,bootstrap_indices,summarize
ROOT=Path(__file__).resolve().parents[1]

def paired(a,b):
    if not np.array_equal(a['date'],b['date']):raise ValueError('Date mismatch')
    if not np.array_equal(a['clean_day'],b['clean_day']):raise ValueError('Mask mismatch')
    ma,mb=metrics(a),metrics(b);n=len(a['date']);full=n==365 and np.all(np.diff(a['date'].astype('datetime64[D]')).astype(int)==1)
    ind=bootstrap_indices(n,7) if full else None
    out={}
    for key in ma:
        diff=mb[key]-ma[key]
        val=summarize(diff,ind) if full else {'mean':float(np.nanmean(diff)),'n_clean_days':int(np.isfinite(diff).sum()),'ci95':None}
        out[key]={'baseline_mean':float(np.nanmean(ma[key])),'dfl_mean':float(np.nanmean(mb[key])),'delta':val}
        if 'peak' in key:out[key]['observed_peak_delta']=float(np.nanmax(mb[key])-np.nanmax(ma[key]))
    return out

def main(args):
    dest=ROOT/'results/analysis';dest.mkdir(exist_ok=True)
    tail='' if args.suffix=='tight' else f'_{args.suffix}'
    records=[]
    for f in sorted((ROOT/'results/network_replay').rglob('dfl_*.json')):
        if f.parent.name in ['main_v1','heldout_v1','main_physical','heldout_physical','code_snapshots']:continue
        st=json.loads(f.read_text());bf=f.with_name(f.name.replace('dfl_','mlp_',1))
        if st.get('parameters',{}).get('dispatch_suffix')!=args.suffix:continue
        if st.get('status')!='completed' or not bf.exists():continue
        bst=json.loads(bf.read_text())
        if bst.get('status')!='completed':continue
        # Pair identity includes every physical scenario field, unlike grouping
        # across unrelated topologies or mappings under the same panel ID.
        for key in ['network','source_pu','mapping_seed','seasonal_weeks','power_factor','period','dispatch_suffix']:
            if st['parameters'][key]!=bst['parameters'][key]:raise ValueError(f'Identity mismatch: {key}')
        with np.load(bf.with_suffix('.npz')) as a,np.load(f.with_suffix('.npz')) as b:
            vals=paired(a,b)
        records.append({'scenario':st['scenario'],'folder':f.parent.name,'parameters':st['parameters'],'outcomes':vals})
    controls=[]
    for name in ['epsilon_low','epsilon_high','night_zero','equal_price','early_tie','late_tie']:
        af=ROOT/f'results/controls{tail}/{name}_mlp.npz';bf=ROOT/f'results/controls{tail}/{name}_dfl.npz'
        if not af.exists() or not bf.exists():continue
        with np.load(af) as a,np.load(bf) as b:
            row={'control':name,'baseline':'mlp','outcomes':paired(a,b),
                'max_charge_difference_kw':float(np.nanmax(abs(a['charge_kw']-b['charge_kw']))),
                'max_discharge_difference_kw':float(np.nanmax(abs(a['discharge_kw']-b['discharge_kw'])))}
            controls.append(row)
        cf=ROOT/f'results/controls{tail}/{name}_msecont.npz'
        if cf.exists():
            with np.load(cf) as a,np.load(bf) as b:controls.append({'control':name,'baseline':'msecont','outcomes':paired(a,b)})
    (dest/f'robustness_pairs{tail}.json').write_text(json.dumps({'regime':args.suffix,'external_mapping_pairs':records,'controls':controls},indent=2),encoding='utf-8')
    mf=ROOT/('results/mechanism' if args.suffix=='tight' else f'results/mechanism_{args.suffix}_vs_msecont');names=['mse','common_only','idiosyncratic_only','dfl']
    if all((mf/f'{x}.npz').exists() for x in names):
        vals={x:metrics(np.load(mf/f'{x}.npz')) for x in names};out={}
        for key in vals['mse']:
            v={x:float(np.nanmean(vals[x][key])) for x in names}
            com=.5*(v['common_only']-v['mse']+v['dfl']-v['idiosyncratic_only'])
            ind=.5*(v['idiosyncratic_only']-v['mse']+v['dfl']-v['common_only'])
            delta=v['dfl']-v['mse']
            if abs(com+ind-delta)>1e-10:raise ValueError('Shapley reconstruction')
            out[key]={'values':v,'common_attribution':com,'idiosyncratic_attribution':ind,'total_difference':delta}
        (dest/f'mechanism_attribution{tail}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'external_mapping_pairs':len(records),'controls':len(controls),'mechanism_complete':all((mf/f'{x}.npz').exists() for x in names)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='tight',choices=['tight','physical']);main(ap.parse_args())
