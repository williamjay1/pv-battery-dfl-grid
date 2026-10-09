"""Exploratory separation of PV information and compensation for load error.

This is an oracle diagnostic, not an implementable policy or a refitted model.
"""
from pathlib import Path
import argparse,json
import numpy as np
from battery_model import Battery,DispatchQP,tariff,realized_bill
from analyze_network import bootstrap_indices,summarize
ROOT=Path(__file__).resolve().parents[1]

def main(args):
    ids=json.loads((ROOT/'datasets/splits_v1.json').read_text())['primary_panels'][0]
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz');hi=[list(a['household_id']).index(x) for x in ids];di=np.flatnonzero(a['split_code']==2)
    gc=a['truth_gc_kw'][hi][:,di];pv=a['truth_pv_kw'][hi][:,di];cl=a['truth_cl_kw'][hi][:,di]
    inputs={};original={}
    for model in ['mlp','msecont','dfl']:
        z=np.load(ROOT/f'results/dispatch/{model}_tou_seed11_test_{args.suffix}.npz');h=[list(z['household_id']).index(x) for x in ids]
        inputs[model]=z['forecast_pv_kw'][h];original[model]=z['bill_aud'][h]
        clean=z['clean_day'][h].all(0);load=z['forecast_load_kw'][h]
    b=Battery();buy,sell=tariff();qp=DispatchQP(tolerance=1e-13);records={};arrays={}
    # The information intervention changes only the load supplied to the frozen
    # optimizer. No forecast is retrained using privileged information.
    for label,lhat in [('shared_forecast',load),('perfect_gc',gc)]:
        for model in ['mlp','msecont','dfl','perfect_pv']:
            phat=pv if model=='perfect_pv' else inputs[model]
            if label=='shared_forecast' and model!='perfect_pv':cost=original[model].copy()
            else:
                cost=np.full((55,365),np.nan)
                for h in range(55):
                    for d in np.flatnonzero(clean):
                        c,dd,_=qp.solve(lhat[h,d]-phat[h,d])
                        cost[h,d]=realized_bill(gc[h,d]-pv[h,d],c,dd,buy,sell,b)+.5*.18*cl[h,d].sum()
            daily=cost.mean(0);daily[~clean]=np.nan;name=label+'__'+model;arrays[name]=daily
            records[name]={'mean_operating_cost_aud':float(np.nanmean(daily)),'n_clean_days':int(clean.sum())}
    idx=bootstrap_indices(365,7)
    for label in ['shared_forecast','perfect_gc']:
        for base in ['mlp','msecont','perfect_pv']:
            name=f'{label}__dfl_minus_{base}'
            records[name]=summarize(arrays[label+'__dfl']-arrays[label+'__'+base],idx)
    shifted_load=load.astype(float)-(inputs['dfl'].astype(float)-inputs['mlp'].astype(float))
    identity=np.nanmax(abs((shifted_load-inputs['mlp'])-(load.astype(float)-inputs['dfl'].astype(float))))
    out={'regime':args.suffix,'status':'exploratory oracle intervention on fixed trained signals; never a deployment benchmark','definition':'A(loadhat-PVhat) is exactly invariant to representing a PV signal correction as the opposite correction to predicted load. Improved operational cost alone cannot identify more accurate PV information.','equivalent_net_identity_max_error_kw':float(identity),'negative_implied_corrected_load_fraction':float(np.mean(shifted_load[:,clean]<0)),'records':records}
    dest=ROOT/'results/analysis';dest.mkdir(exist_ok=True)
    (dest/f'signal_semantics_{args.suffix}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    np.savez_compressed(dest/f'signal_semantics_{args.suffix}_daily.npz',date=a['date'][di],clean_day=clean,**arrays)
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='tight',choices=['tight','physical']);main(ap.parse_args())
