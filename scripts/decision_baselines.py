"""Frozen low-dimensional cost calibration and joint residual-path SAA.

No test label is used for fitting or choosing either method. Calibration fits
three fixed daylight-bin biases to training bills and selects shrinkage on
validation bills. SAA uses 20 paired GC/PV validation residual days shared
across households; all paths share the same battery action vector.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
import torch
from battery_model import Battery,DispatchQP,tariff,realized_bill
from decision_layer import BatteryLayer
from evaluate_dispatch import selected_households

ROOT=Path(__file__).resolve().parents[1]
HOURS=np.arange(48)*.5
BANDS=np.stack([(HOURS>=6)&(HOURS<10),(HOURS>=10)&(HOURS<14),(HOURS>=14)&(HOURS<18)]).astype(float)

def bill_tensor(actual,c,d,kind):
    buy,sell=tariff(kind);g=actual+c-d
    return .5*(torch.as_tensor(buy)*torch.relu(g)+torch.as_tensor(sell)*torch.minimum(g,torch.zeros_like(g))+.02*(c+d)).sum(-1)

def prepare(a,splits):
    ids=selected_households(splits);hs=np.array([int(np.flatnonzero(a['household_id']==i)[0]) for i in ids])
    clean=a['valid_history']&a['valid_target_gcgg']
    known=a['household_role']==0
    train=np.argwhere(clean&known[:,None]&(a['split_code']==0)[None,:])
    val=np.argwhere(clean&known[:,None]&(a['split_code']==1)[None,:])
    train=train[np.random.default_rng(7812).choice(len(train),min(2048,len(train)),replace=False)]
    val=val[np.random.default_rng(7813).choice(len(val),min(2048,len(val)),replace=False)]
    return ids,hs,train,val

def fit_calibration(a,train,val,kind):
    start=time.perf_counter();torch.manual_seed(7814);raw=torch.nn.Parameter(torch.zeros(3,dtype=torch.float64));optimizer=torch.optim.Adam([raw],lr=.03);layer=BatteryLayer(kind=kind)
    bands=torch.tensor(BANDS);trace=[]
    th,td=train.T;vh,vd=val.T
    tr_load=torch.tensor(a['gc_shared'][th,td],dtype=torch.float64);tr_pv=torch.tensor(a['pv_mlp_seed11'][th,td],dtype=torch.float64);tr_cap=torch.tensor(a['capacity_kwp'][th,td,None],dtype=torch.float64)
    actual=torch.tensor(a['truth_gc_kw'][th,td]-a['truth_pv_kw'][th,td],dtype=torch.float64)
    rng=np.random.default_rng(7815)
    for epoch in range(3):
        perm=rng.permutation(len(train));values=[]
        for start_batch in range(0,len(train),32):
            ix=perm[start_batch:start_batch+32];bias=.25*torch.tanh(raw)
            pv=torch.clamp(tr_pv[ix]+tr_cap[ix]*(bias@bands),min=0)
            c,d,e=layer(tr_load[ix]-pv);loss=bill_tensor(actual[ix],c,d,kind).mean();optimizer.zero_grad();loss.backward();optimizer.step();values.append(float(loss.detach()))
        trace.append({'epoch':epoch+1,'train_sample_mean_bill_aud':float(np.mean(values)),'bias_kw_per_kwp':(.25*torch.tanh(raw)).detach().tolist()});print(f'Calibration {kind} epoch {epoch+1}: {trace[-1]}',flush=True)
    beta=(.25*torch.tanh(raw)).detach().numpy();qp=DispatchQP(kind=kind,tolerance=1e-13);buy,sell=tariff(kind);candidates=[]
    for alpha in [0,.25,.5,.75,1]:
        pv=np.maximum(a['pv_mlp_seed11'][vh,vd]+a['capacity_kwp'][vh,vd,None]*(alpha*beta@BANDS),0);bills=[]
        for i,(h,d) in enumerate(val):
            c,dd,_=qp.solve(a['gc_shared'][h,d]-pv[i]);bills.append(float(realized_bill(a['truth_gc_kw'][h,d]-a['truth_pv_kw'][h,d],c,dd,buy,sell)))
        candidates.append({'alpha':alpha,'mean_validation_bill_aud':float(np.mean(bills))})
    chosen=int(np.argmin([c['mean_validation_bill_aud'] for c in candidates]));alpha=candidates[chosen]['alpha']
    return {'tariff':kind,'time_bins':['06:00-10:00','10:00-14:00','14:00-18:00'],'base_model':'MSE seed11','train_household_days':len(train),'validation_household_days':len(val),'training_date_range':[str(a['date'][td.min()]),str(a['date'][td.max()])],'validation_date_range':[str(a['date'][vd.min()]),str(a['date'][vd.max()])],'training_ids':np.unique(a['household_id'][th]).tolist(),'validation_ids':np.unique(a['household_id'][vh]).tolist(),'train_bias_kw_per_kwp':beta.tolist(),'selected_alpha':alpha,'selected_bias_kw_per_kwp':(alpha*beta).tolist(),'trace':trace,'validation_candidates':candidates,'test_used_for_fitting':False,'elapsed_seconds':time.perf_counter()-start}

def residual_bank(a,hs):
    known=np.flatnonzero(a['household_role']==0)
    mask=(a['valid_history'][known]&a['valid_target_gcgg'][known]).all(0)&(a['split_code']==1)
    eligible=np.flatnonzero(mask)
    if len(eligible)<20:raise ValueError('Fewer than 20 jointly clean validation residual dates')
    ds=eligible[np.linspace(0,len(eligible)-1,20,dtype=int)]
    donor=hs.copy();held=np.flatnonzero(a['household_role'][hs]==1)
    ranked=sorted(known,key=lambda h:hashlib.sha256(f'residual-donor-v1:{int(a["household_id"][h])}'.encode()).hexdigest())
    for j,i in enumerate(held):donor[i]=ranked[j]
    gc=a['truth_gc_kw'][donor[:,None],ds[None,:]]-a['gc_shared'][donor[:,None],ds[None,:]]
    pv=(a['truth_pv_kw'][donor[:,None],ds[None,:]]-a['pv_mlp_seed11'][donor[:,None],ds[None,:]])/a['capacity_kwp'][donor[:,None],ds[None,:],None]
    return gc,pv,{'n_scenarios':20,'eligible_joint_clean_validation_dates':len(eligible),'scenario_dates':a['date'][ds].tolist(),'donor_household_ids':a['household_id'][donor].tolist(),'target_household_ids':a['household_id'][hs].tolist(),'path_construction':'Paired GC and capacity-normalized PV residuals from the same observed validation day; scenario date shared across all households; marginal scenarios clipped to nonnegative physical load/PV','known_households_use_own_residuals':True,'heldout_residual_policy':'Deterministic hash-ranked known donor without replacement; no heldout actual target used to estimate residuals','network_geography_claim':False,'test_used_for_fitting':False}

def seasonal_weeks(dates):
    chosen=[];dt=dates.astype('datetime64[D]');weekday=(dt.astype(int)+3)%7
    for month in ['2012-07','2012-10','2013-01','2013-04']:
        start=np.flatnonzero(np.char.startswith(dates,month)&(weekday==0))[0];chosen.extend(range(start,start+7))
    return np.array(chosen)

def replay(a,ids,hs,kind,method,parameters,seasonal=False,limit_days=0):
    start=time.perf_counter();ds=np.flatnonzero(a['split_code']==2)
    if seasonal:ds=ds[seasonal_weeks(a['date'][ds])]
    if limit_days:ds=ds[:limit_days]
    dates=a['date'][ds];H=len(hs);D=len(ds);gc=a['truth_gc_kw'][hs][:,ds];pv=a['truth_pv_kw'][hs][:,ds];cl=a['truth_cl_kw'][hs][:,ds]
    load=a['gc_shared'][hs][:,ds];forecast=a['pv_mlp_seed11'][hs][:,ds].copy();valid=a['valid_history'][hs][:,ds]&a['valid_target_feeder'][hs][:,ds]
    inputfinite=a['valid_input_finite'][hs][:,ds];cap=a['capacity_kwp'][hs][:,ds]
    if method=='calibrated':forecast=np.maximum(forecast+cap[:,:,None]*(np.array(parameters['selected_bias_kw_per_kwp'])@BANDS),0)
    else:rg,rp,bank=parameters
    battery=Battery();buy,sell=tariff(kind);qp=DispatchQP(battery,kind,n_scenarios=20 if method=='stochastic' else 1,tolerance=1e-13)
    charge=np.full_like(gc,np.nan);discharge=np.full_like(gc,np.nan);end=np.full((H,D),np.nan,np.float32);bills=np.full((H,D),np.nan);failures=[];maxbalance=0;maxsim=0
    for i in range(H):
        for j in range(D):
            if not inputfinite[i,j] or not np.isfinite(load[i,j]).all() or not np.isfinite(forecast[i,j]).all():continue
            try:
                net=load[i,j]-forecast[i,j]
                if method=='stochastic':net=np.maximum(load[i,j][None,:]+rg[i],0)-np.maximum(forecast[i,j][None,:]+rp[i]*cap[i,j],0)
                c,d,e,diag=qp.solve(net,True);charge[i,j]=c;discharge[i,j]=d;end[i,j]=e[-1]
                maxbalance=max(maxbalance,diag['energy_balance_max']);maxsim=max(maxsim,diag['max_simultaneous_kw'])
                bills[i,j]=realized_bill(gc[i,j]-pv[i,j],c,d,buy,sell,battery)+.5*.18*np.sum(cl[i,j])
            except Exception as exc:failures.append({'household_id':int(ids[i]),'date':str(dates[j]),'error':str(exc)})
        if (i+1)%25==0:print(f'{method}/{kind}: {i+1}/{H} households, {D} dates, {time.perf_counter()-start:.1f}s',flush=True)
    folder=ROOT/'results/dispatch';folder.mkdir(exist_ok=True)
    stem=f'{method}_{kind}_seed11_test_tight'+('_seasonal' if seasonal else '')+('_benchmark' if limit_days else '')
    target=folder/f'{stem}.npz'
    np.savez_compressed(target,household_id=ids,date=dates,charge_kw=charge,discharge_kw=discharge,bill_aud=bills,end_energy_kwh=end,clean_day=valid,actual_load_kw=gc+cl,actual_pv_kw=pv,forecast_load_kw=load,forecast_pv_kw=forecast)
    report={'method':method,'tariff':kind,'seed':11,'n_households':H,'n_dates':D,'date_range':[str(dates[0]),str(dates[-1])],'dates':dates.tolist(),'clean_household_days':int(valid.sum()),'failed':len(failures),'failures':failures,'max_energy_balance_residual':maxbalance,'max_simultaneous_kw':maxsim,'max_terminal_error_kwh':float(np.nanmax(abs(end-5))),'mean_clean_bill_aud':float(np.nanmean(bills[valid])),'elapsed_seconds':time.perf_counter()-start,'solver_tolerance':1e-13,'test_used_for_fitting':False,'actual_measurements_used_only_after_actions_for_settlement':True,'output':str(target)}
    (folder/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if failures:raise RuntimeError('Failed QP samples retained; resolve before final analysis')
    return report

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    z=np.load(ROOT/'datasets/forecast_arrays_v1.npz');keys=['household_id','household_role','date','split_code','valid_history','valid_target_gcgg','valid_target_feeder','valid_input_finite','gc_shared','pv_mlp_seed11','capacity_kwp','truth_gc_kw','truth_pv_kw','truth_cl_kw'];a={k:z[k] for k in keys}
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text());ids,hs,train,val=prepare(a,splits)
    if args.method=='calibrated':
        path=ROOT/f'results/decision_baselines_calibrated_{args.tariff}.json'
        if path.exists():parameters=json.loads(path.read_text())
        else:parameters=fit_calibration(a,train,val,args.tariff);path.write_text(json.dumps(parameters,indent=2))
    else:
        parameters=residual_bank(a,hs);(ROOT/'results/decision_baselines_stochastic_bank.json').write_text(json.dumps(parameters[2],indent=2))
    replay(a,ids,hs,args.tariff,args.method,parameters,args.seasonal,args.limit_days)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['calibrated','stochastic'],required=True);ap.add_argument('--tariff',choices=['tou','flat'],required=True);ap.add_argument('--seasonal',action='store_true');ap.add_argument('--limit-days',type=int,default=0);main(ap.parse_args())
