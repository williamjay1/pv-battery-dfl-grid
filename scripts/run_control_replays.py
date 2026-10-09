"""Predeclared dispatch-selection, nighttime-signal and equal-price controls."""
from pathlib import Path
import argparse,json,time
import numpy as np
from battery_model import Battery,DispatchQP,tariff,realized_bill
from network_model import EuropeanLV

ROOT=Path(__file__).resolve().parents[1]

def main(args):
    ids=json.loads((ROOT/'datasets/splits_v1.json').read_text())['primary_panels'][0]
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    ha=[list(a['household_id']).index(i) for i in ids];di=np.flatnonzero(a['split_code']==2)
    gc=a['truth_gc_kw'][ha][:,di];cl=a['truth_cl_kw'][ha][:,di];pv=a['truth_pv_kw'][ha][:,di]
    load=gc+cl;network=EuropeanLV();dest=ROOT/'results'/('controls' if args.suffix=='tight' else f'controls_{args.suffix}');dest.mkdir(exist_ok=True)
    plan={'epsilon_low':(1e-5,False,'tou'),'epsilon_high':(1e-3,False,'tou'),
          'night_zero':(1e-4,True,'tou'),'equal_price':(1e-4,False,'equal'),
          'early_tie':(1e-4,False,'tou'),'late_tie':(1e-4,False,'tou')}
    for label in args.controls:
        eps,night,kind=plan[label];b=Battery(epsilon=eps);buy,sell=tariff(kind)
        for model in args.models:
            path=dest/f'{label}_{model}.npz';statuspath=path.with_suffix('.json')
            if path.exists() and statuspath.exists():
                old=json.loads(statuspath.read_text())
                if old['status']=='complete':print('reuse',path.name,flush=True);continue
            start=time.perf_counter();src=np.load(ROOT/f'results/dispatch/{model}_tou_seed11_test_{args.suffix}.npz')
            hs=[list(src['household_id']).index(i) for i in ids]
            pred=src['forecast_pv_kw'][hs].copy();loadpred=src['forecast_load_kw'][hs]
            if night:pred[:,:,:8]=0;pred[:,:,42:]=0
            c=np.full_like(pred,np.nan);d=np.full_like(pred,np.nan);bill=np.full(pred.shape[:2],np.nan)
            perturbation=None
            if label in ['early_tie','late_tie']:
                direction=1 if label=='early_tie' else -1
                perturbation=b.epsilon*np.r_[direction*np.linspace(-1,1,48)/b.power,np.zeros(48)]
            qp=DispatchQP(b,kind,tolerance=1e-13,perturbation=perturbation)
            equal_actions=qp.solve(np.zeros(48)) if kind=='equal' else None
            for h in range(55):
                for j in range(len(di)):
                    if not np.isfinite(pred[h,j]).all() or not np.isfinite(loadpred[h,j]).all():continue
                    cc,dd,_=equal_actions if equal_actions is not None else qp.solve(loadpred[h,j]-pred[h,j])
                    c[h,j]=cc;d[h,j]=dd
                    bill[h,j]=realized_bill(gc[h,j]-pv[h,j],cc,dd,buy,sell,b)+.5*.18*cl[h,j].sum()
            trans=lambda x:x.transpose(1,2,0).reshape(-1,55)
            out=network.run(trans(load),trans(pv),trans(c-d),return_bus=True)
            if np.any(out['input_valid']&~out['converged']):raise ValueError('AC failure retained')
            arrays={k:v for k,v in out.items() if isinstance(v,np.ndarray)}
            vh=out['household_voltage_pu']
            wider=100*((vh<.9)|(vh>1.1)).mean(1);wider[~out['converged']]=np.nan
            wear=.5*.02*(c+d).sum(2)
            arrays.update(date=src['date'],household_id=np.array(ids),clean_day=src['clean_day'][hs].all(0),
                bill_mean_aud=bill.mean(0),bill_byhouse_aud=bill.T,charge_kw=c,discharge_kw=d,
                retail_bill_mean_aud=(bill-wear).mean(0),throughput_cost_mean_aud=wear.mean(0),
                converged_day=out['converged'].reshape(-1,48).all(1),voltage_violation_fraction_090_110=wider/100)
            np.savez_compressed(path,**arrays)
            status={'status':'complete','control':label,'model':model,'battery':b.__dict__,'night_zero':night or args.suffix=='physical','tariff':kind,'regime':args.suffix,'n_clean_days':int(arrays['clean_day'].sum()),'elapsed_seconds':time.perf_counter()-start,'test_used_for_fitting':False}
            if perturbation is not None:status['selection_sensitivity']='Frozen forecasts transported to alternative common deterministic selection: epsilon times +/- linearly increasing time weight on normalized charging, in addition to the same strictly convex regularizer. This is not retraining under a new controller.'
            statuspath.write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--controls',nargs='+',default=['epsilon_low','epsilon_high','night_zero','equal_price']);ap.add_argument('--models',nargs='+',default=['mlp','dfl']);ap.add_argument('--suffix',default='tight',choices=['tight','physical'])
    main(ap.parse_args())
