"""Paired calendar-block inference for frozen network replay outputs."""
from pathlib import Path
import argparse,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def metrics(z):
    days=len(z['date']);clean=z['clean_day']&z['converged_day']
    def slots(k):return z[k].reshape(days,48)
    imp=slots('source_import_kw')
    if 'voltage_violation_fraction_090_110' in z.files:
        wider=slots('voltage_violation_fraction_090_110')
    else:
        vh=z['household_voltage_pu'];wider=((vh<.9)|(vh>1.1)).mean(1).reshape(days,48)
    out={'bill_aud':z['bill_mean_aud'].astype(float),
         'voltage_percentage_points':100*slots('voltage_violation_fraction').mean(1),
         'mean_voltage_excursion_pu':slots('voltage_exceedance_pu').mean(1)/55,
         'daily_import_peak_kw':np.maximum(imp,0).max(1),
         'daily_export_peak_kw':np.maximum(-imp,0).max(1),
         'loss_kwh':.5*slots('loss_kw').sum(1),
         'voltage_090_110_percentage_points':100*wider.mean(1)}
    for value in out.values():value[~clean]=np.nan
    return out

def bootstrap_indices(n,block,reps=2000,seed=20261005):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]

def summarize(x,index):
    x=np.asarray(x);res=np.nanmean(x[index],axis=1)
    return {'mean':float(np.nanmean(x)), 'ci95':np.quantile(res,[.025,.975]).tolist(),
            'n_clean_days':int(np.isfinite(x).sum())}

def main(args):
    if args.run_tag not in ['main_v1','heldout_v1','main_physical','heldout_physical']:
        raise ValueError('This three-seed grouping is restricted to fixed main/heldout scenarios; use analyze_robustness.py for topology/mapping pairs')
    src=ROOT/'results/network_replay'/args.run_tag
    dest=ROOT/'results/analysis';dest.mkdir(exist_ok=True)
    cases={}
    for f in sorted(src.glob('*.json')):
        st=json.loads(f.read_text(encoding='utf-8'))
        if st.get('status')!='completed' or st.get('objective') not in [args.baseline,'dfl']:continue
        if st.get('solver_failed_slots'):raise ValueError(f'Solver failures: {f}')
        key=(st['panel'],st['battery_count'],st['tariff_mix'])
        # Infer adoption from status parameter in older compatible reports.
        cases.setdefault(key,{})[(st['objective'],st['seed'])]=(f.with_suffix('.npz'),st)
    records=[];daily={};pending=[]
    for key,files in sorted(cases.items()):
        need=[(m,s) for m in [args.baseline,'dfl'] for s in [11,23,47]]
        if any(k not in files for k in need):pending.append(list(key));continue
        arrays={};dates=None
        for k,(file,st) in files.items():
            z=np.load(file)
            if dates is None:dates=z['date']
            elif not np.array_equal(dates,z['date']):raise ValueError('Unmatched dates')
            arrays[k]=metrics(z)
        n=len(dates)
        if n!=365 or not np.all(np.diff(dates.astype('datetime64[D]')).astype(int)==1):raise ValueError('Expected full 365-day calendar axis')
        idx={b:bootstrap_indices(n,b,args.reps) for b in [7,14,28]}
        rec={'panel':key[0],'battery_count':key[1],'tariff_mix':key[2],'outcomes':{}}
        for metric in arrays[(args.baseline,11)]:
            base=np.stack([arrays[(args.baseline,s)][metric] for s in [11,23,47]])
            new=np.stack([arrays[('dfl',s)][metric] for s in [11,23,47]])
            if not np.array_equal(np.isfinite(base),np.isfinite(new)):raise ValueError('Unpaired masks')
            x=(new-base).mean(0)
            rec['outcomes'][metric]={'mse':float(np.nanmean(base)),'dfl':float(np.nanmean(new)),
              'delta_dfl_minus_mse':summarize(x,idx[7]),
              'block_sensitivity':{str(b):summarize(x,idx[b])['ci95'] for b in [14,28]},
              'seed_differences':[float(np.nanmean(v)) for v in new-base]}
            if 'peak' in metric:
                rec['outcomes'][metric]['observed_period_peaks_by_seed']={m:[float(np.nanmax(arrays[(m,s)][metric])) for s in [11,23,47]] for m in [args.baseline,'dfl']}
            daily[f'p{key[0]}_b{key[1]}_{key[2]}_{metric}']=x
        records.append(rec)
        cost=rec['outcomes']['bill_aud']['delta_dfl_minus_mse']
        factor=55/key[1]
        rec['operating_cost_delta_per_battery_owner_aud']={'mean':cost['mean']*factor,'ci95':[v*factor for v in cost['ci95']],'denominator_battery_owners':key[1]}
    report={'run_tag':args.run_tag,'baseline':args.baseline,'complete_paired_cells':len(records),'pending':pending,
      'inference':'Conditional on fixed benchmark panels; calendar-block percentile intervals, training seeds averaged, no iid half-hour inference',
      'replicates':args.reps,'records':records}
    (dest/f'{args.run_tag}_vs_{args.baseline}_paired.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if daily:np.savez_compressed(dest/f'{args.run_tag}_vs_{args.baseline}_paired_daily.npz',**daily)
    print(json.dumps({'run_tag':args.run_tag,'baseline':args.baseline,'complete_paired_cells':len(records),'pending':len(pending)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-tag',default='main_v1');ap.add_argument('--reps',type=int,default=2000);ap.add_argument('--baseline',default='mlp',choices=['mlp','msecont'])
    main(ap.parse_args())
