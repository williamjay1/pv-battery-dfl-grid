"""Pre-specified common/idiosyncratic decomposition of task-signal shifts."""
from pathlib import Path
import json,time,argparse
import numpy as np
from battery_model import Battery,DispatchQP,realized_bill,tariff
from network_model import EuropeanLV

ROOT=Path(__file__).resolve().parents[1]

def main(args):
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text())
    ids=splits['primary_panels'][0]
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    m=np.load(ROOT/f'results/dispatch/{args.baseline}_tou_seed11_test_{args.suffix}.npz')
    d=np.load(ROOT/f'results/dispatch/dfl_tou_seed11_test_{args.suffix}.npz')
    hi=[list(m['household_id']).index(i) for i in ids]
    ha=[list(a['household_id']).index(i) for i in ids]
    di=np.flatnonzero(a['split_code']==2)
    cap=a['capacity_kwp'][ha][:,di-1,None]
    base=m['forecast_pv_kw'][hi];full=d['forecast_pv_kw'][hi]
    correction=(full-base)/cap
    common=np.mean(correction,axis=0,keepdims=True)
    residual=correction-common
    inputs={'mse':base,'common_only':np.maximum(0,base+cap*common),
            'idiosyncratic_only':np.maximum(0,base+cap*residual),'dfl':full}
    if np.nanmax(abs(base+cap*(common+residual)-full))>1e-5:raise ValueError('Decomposition does not reconstruct')
    load=m['actual_load_kw'][hi];pv=m['actual_pv_kw'][hi]
    gc=a['truth_gc_kw'][ha][:,di];cl=a['truth_cl_kw'][ha][:,di]
    loadforecast=m['forecast_load_kw'][hi];clean=m['clean_day'][hi].all(0)
    dirname='mechanism' if args.suffix=='tight' and args.baseline=='mlp' else f'mechanism_{args.suffix}_vs_{args.baseline}'
    dest=ROOT/'results'/dirname;dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'signal_components.npz',date=m['date'],household_id=np.array(ids),common_normalized=common,idiosyncratic_normalized=residual,clean_day=clean)
    qp=DispatchQP(tolerance=1e-13);buy,sell=tariff();b=Battery();network=EuropeanLV()
    reports={}
    for name,pred in inputs.items():
        start=time.perf_counter()
        if name in ['mse','dfl']:
            src=m if name=='mse' else d;c=src['charge_kw'][hi];dd=src['discharge_kw'][hi];bill=src['bill_aud'][hi]
        else:
            c=np.full_like(pred,np.nan);dd=np.full_like(pred,np.nan);bill=np.full(pred.shape[:2],np.nan)
            for h in range(55):
                for j in range(len(di)):
                    if not np.isfinite(pred[h,j]).all() or not np.isfinite(loadforecast[h,j]).all():continue
                    c[h,j],dd[h,j],_=qp.solve(loadforecast[h,j]-pred[h,j])
                    bill[h,j]=realized_bill(gc[h,j]-pv[h,j],c[h,j],dd[h,j],buy,sell,b)+.5*.18*cl[h,j].sum()
        transform=lambda x:x.transpose(1,2,0).reshape(-1,55)
        out=network.run(transform(load),transform(pv),transform(c-dd),return_bus=True)
        if np.any(out['input_valid']&~out['converged']):raise ValueError('AC failure')
        net=(c-dd).sum(0)
        arrays={k:v for k,v in out.items() if isinstance(v,np.ndarray)}
        arrays.update(date=m['date'],clean_day=clean,bill_byhouse_aud=bill.T,bill_mean_aud=bill.mean(0),
          charge_kw=c,discharge_kw=dd,converged_day=out['converged'].reshape(-1,48).all(1),
          aggregate_battery_kw=net,aggregate_battery_squared_kw2=np.mean(net**2,axis=1))
        np.savez_compressed(dest/f'{name}.npz',**arrays)
        reports[name]={'elapsed_seconds':time.perf_counter()-start,'clean_days':int(clean.sum()),'input_valid_slots':int(out['input_valid'].sum()),'ac_failures':int(np.sum(out['input_valid']&~out['converged']))}
        print(name,reports[name],flush=True)
    reports['definition']='Exact two-component counterfactual signal decomposition; clips negative constructed PV signals; no causal attribution to real household behavior'
    reports['baseline']=args.baseline;reports['regime']=args.suffix
    (dest/'replay.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='tight',choices=['tight','physical']);ap.add_argument('--baseline',default='mlp',choices=['mlp','msecont']);main(ap.parse_args())
