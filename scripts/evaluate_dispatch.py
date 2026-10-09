"""Physical replay and true settlement of frozen forecasts, without test fitting."""
from pathlib import Path
import argparse,json,time
import numpy as np
from battery_model import Battery,DispatchQP,tariff,realized_bill,self_consumption

ROOT=Path(__file__).resolve().parents[1]

def selected_households(splits):
    panels=splits['primary_panels']
    if isinstance(panels,dict):panels=list(panels.values())
    ids=[]
    for group in panels:
        if isinstance(group,dict):group=group.get('household_ids',group.get('ids'))
        ids.extend(group)
    ids.extend(splits['heldout_network_household_ids'])
    return np.array(list(dict.fromkeys(ids)),dtype=int)

def main(args):
    start=time.perf_counter();splits=json.loads((ROOT/'datasets/splits_v1.json').read_text(encoding='utf-8'))
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    if args.households=='all_known':ids=a['household_id'][a['household_role']==0]
    else:ids=selected_households(splits)
    hs=np.array([int(np.flatnonzero(a['household_id']==i)[0]) for i in ids])
    ds=np.flatnonzero(a['split_code']==(1 if args.period=='validation' else 2))
    if args.days:ds=ds[:args.days]
    dates=a['date'][ds];H=len(hs);D=len(ds)
    battery=Battery(epsilon=args.epsilon,capacity=args.capacity,power=args.power,throughput=args.throughput)
    buy,sell=tariff(args.tariff)
    gc=a['truth_gc_kw'][hs][:,ds];pv=a['truth_pv_kw'][hs][:,ds];cl=a['truth_cl_kw'][hs][:,ds]
    valid=a['valid_history'][hs][:,ds]&a['valid_target_feeder'][hs][:,ds]
    inputfinite=a['valid_input_finite'][hs][:,ds]
    load=a['gc_shared'][hs][:,ds]
    if args.model=='msecont':
        da=np.load(ROOT/f'datasets/mse_continuation_arrays_{args.mse_tag}.npz')
        forecast=da[f'pv_msecont_{args.tariff}_seed{args.seed}'][hs][:,ds]
    elif args.model.startswith('dfl'):
        da=np.load(ROOT/f'datasets/decision_arrays_{args.decision_tag}.npz')
        forecast=da[f'pv_dfl_{args.tariff}_seed{args.seed}'][hs][:,ds]
    elif args.model=='mlp':forecast=a[f'pv_mlp_seed{args.seed}'][hs][:,ds]
    elif args.model=='oracle':forecast=pv.copy();load=gc.copy()
    elif args.model in ['none','self']:forecast=pv.copy()
    else:forecast=a[f'pv_{args.model}'][hs][:,ds]
    if args.night_zero:
        forecast=forecast.copy();forecast[:,:,:8]=0;forecast[:,:,42:]=0
    charge=np.full_like(gc,np.nan);discharge=np.full_like(gc,np.nan)
    end_energy=np.full((H,D),np.nan,dtype=np.float32);bills=np.full((H,D),np.nan,dtype=np.float64)
    nfailed=0;maxbalance=0;maxsimultaneous=0;failures=[]
    qp=DispatchQP(battery,args.tariff,tolerance=1e-13)
    for i in range(H):
        if args.model=='self':
            if not np.isfinite(gc[i]-pv[i]).all():raise ValueError('Self-consumption continuous path cannot bridge missing actual data')
            cc,dd,last=self_consumption(gc[i]-pv[i],battery)
            charge[i]=cc;discharge[i]=dd
            end_energy[i]=battery.initial_fraction*battery.capacity+np.cumsum(battery.dt*(battery.efficiency*cc-dd/battery.efficiency).sum(1))
        for j in range(D):
            if not inputfinite[i,j] or not np.isfinite(forecast[i,j]).all() or not np.isfinite(load[i,j]).all():continue
            try:
                if args.model=='none':c=np.zeros(48);d=c.copy();e=np.full(48,battery.initial_fraction*battery.capacity)
                elif args.model=='self':c=charge[i,j];d=discharge[i,j];e=np.full(48,end_energy[i,j])
                else:
                    c,d,e,diag=qp.solve(load[i,j]-forecast[i,j],True)
                    maxbalance=max(maxbalance,diag['energy_balance_max']);maxsimultaneous=max(maxsimultaneous,diag['max_simultaneous_kw'])
                charge[i,j]=c;discharge[i,j]=d;end_energy[i,j]=e[-1]
                # CL is separately metered and never battery-offset. Its fixed
                # retail price is a transparent assumption, constant across arms.
                bills[i,j]=realized_bill(gc[i,j]-pv[i,j],c,d,buy,sell,battery)+.5*.18*np.sum(cl[i,j])
            except Exception as exc:
                nfailed+=1;failures.append({'household':int(ids[i]),'date':str(dates[j]),'error':str(exc)})
        if (i+1)%25==0:print(f'{args.model}/{args.tariff}/{args.seed} households {i+1}/{H}, {time.perf_counter()-start:.1f}s',flush=True)
    dest=ROOT/'results/dispatch';dest.mkdir(exist_ok=True)
    suffix=args.suffix or 'base'
    stem=f'{args.model}_{args.tariff}_seed{args.seed}_{args.period}_{suffix}'
    np.savez_compressed(dest/f'{stem}.npz',household_id=ids,date=dates,charge_kw=charge,discharge_kw=discharge,bill_aud=bills,end_energy_kwh=end_energy,clean_day=valid,actual_load_kw=gc+cl,actual_pv_kw=pv,forecast_pv_kw=forecast,forecast_load_kw=load)
    report={'model':args.model,'tariff':args.tariff,'seed':args.seed,'period':args.period,'battery':battery.__dict__,'solver_tolerance':1e-13,'n_households':H,'n_dates':D,'clean_household_days':int(valid.sum()),'n_failed':nfailed,'failures':failures,'max_energy_balance_residual':maxbalance,'max_simultaneous_charge_discharge_kw':maxsimultaneous,'elapsed_seconds':time.perf_counter()-start,'mean_clean_bill_aud':float(np.nanmean(bills[valid])),'test_used_for_fitting':False,'night_zero':args.night_zero,'output':str(dest/f'{stem}.npz')}
    (dest/f'{stem}.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if nfailed:raise RuntimeError(f'{nfailed} failed cases retained; resolve before analysis')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['mlp','dfl','msecont','hgb','persistence','weekly','oracle','none','self']);ap.add_argument('--tariff',default='tou',choices=['tou','flat','equal']);ap.add_argument('--seed',type=int,default=11);ap.add_argument('--decision-tag',default='v1');ap.add_argument('--mse-tag',default='v1');ap.add_argument('--period',default='test',choices=['test','validation']);ap.add_argument('--households',default='panels',choices=['panels','all_known']);ap.add_argument('--epsilon',type=float,default=1e-4);ap.add_argument('--capacity',type=float,default=10);ap.add_argument('--power',type=float,default=3);ap.add_argument('--throughput',type=float,default=.02);ap.add_argument('--days',type=int,default=0);ap.add_argument('--night-zero',action='store_true');ap.add_argument('--suffix',default='')
    main(ap.parse_args())
