"""Original calibration/SAA design with physical night support in every path."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from decision_baselines import prepare,fit_calibration,residual_bank,BANDS
from battery_model import Battery,DispatchQP,tariff,realized_bill
from train_decision_physical import ROOT,NIGHT_SLOTS,REDESIGN

def main(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    z=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    keys=['household_id','household_role','date','split_code','valid_history','valid_target_gcgg','valid_target_feeder','valid_input_finite','gc_shared','pv_mlp_seed11','capacity_kwp','truth_gc_kw','truth_pv_kw','truth_cl_kw']
    a={k:z[k] for k in keys};a['pv_mlp_seed11']=a['pv_mlp_seed11'].copy()
    # Only finite predictions are transformed; unavailable histories remain NaN.
    finite=np.isfinite(a['pv_mlp_seed11']).all(2)
    for slot in NIGHT_SLOTS:a['pv_mlp_seed11'][:,:,slot][finite]=0
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text());ids,hs,tr,va=prepare(a,splits)
    kind=args.tariff;method=args.method
    if method=='calibrated':
        path=ROOT/f'results/decision_baselines_physical_calibrated_{kind}.json'
        if path.exists():parameters=json.loads(path.read_text())
        else:
            parameters=fit_calibration(a,tr,va,kind)
            parameters.update(redesign_disclosure=REDESIGN,night_zero_slots=NIGHT_SLOTS,night_support_in_training_validation_inference=True)
            path.write_text(json.dumps(parameters,indent=2))
    else:
        rg,rp,parameters=residual_bank(a,hs)
        # Every scenario has zero night PV, including the held-out donor paths.
        rp[:,:,NIGHT_SLOTS]=0
        parameters.update(redesign_disclosure=REDESIGN,night_zero_slots=NIGHT_SLOTS,night_support_applied_to_every_scenario=True,
                          cross_household_scenario_alignment='Retained for provenance but has no coordinating effect in these separable household optimizations')
        (ROOT/f'results/decision_baselines_physical_stochastic_bank_{kind}.json').write_text(json.dumps(parameters,indent=2))
    ds=np.flatnonzero(a['split_code']==2);dates=a['date'][ds];H=len(hs);D=len(ds)
    gc=a['truth_gc_kw'][hs][:,ds];pv=a['truth_pv_kw'][hs][:,ds];cl=a['truth_cl_kw'][hs][:,ds]
    load=a['gc_shared'][hs][:,ds];forecast=a['pv_mlp_seed11'][hs][:,ds].copy();cap=a['capacity_kwp'][hs][:,ds]
    if method=='calibrated':forecast=np.maximum(forecast+cap[:,:,None]*(np.asarray(parameters['selected_bias_kw_per_kwp'])@BANDS),0)
    valid=a['valid_history'][hs][:,ds]&a['valid_target_feeder'][hs][:,ds];inputfinite=a['valid_input_finite'][hs][:,ds]
    battery=Battery();buy,sell=tariff(kind);qp=DispatchQP(battery,kind,n_scenarios=20 if method=='stochastic' else 1,tolerance=1e-13)
    charge=np.full_like(gc,np.nan);discharge=np.full_like(gc,np.nan);end=np.full((H,D),np.nan,np.float32);bills=np.full((H,D),np.nan)
    failures=[];maxbalance=0.;maxsim=0.;maxnight=0.;start=time.perf_counter()
    for i in range(H):
        for j in range(D):
            if not inputfinite[i,j] or not np.isfinite(load[i,j]).all() or not np.isfinite(forecast[i,j]).all():continue
            try:
                pathpv=forecast[i,j]
                if method=='stochastic':
                    pathpv=np.maximum(forecast[i,j][None,:]+rp[i]*cap[i,j],0);pathpv[:,NIGHT_SLOTS]=0
                    net=np.maximum(load[i,j][None,:]+rg[i],0)-pathpv
                else:net=load[i,j]-pathpv
                maxnight=max(maxnight,float(np.max(abs(pathpv[...,NIGHT_SLOTS]))))
                c,d,e,diag=qp.solve(net,True);charge[i,j]=c;discharge[i,j]=d;end[i,j]=e[-1]
                maxbalance=max(maxbalance,diag['energy_balance_max']);maxsim=max(maxsim,diag['max_simultaneous_kw'])
                bills[i,j]=realized_bill(gc[i,j]-pv[i,j],c,d,buy,sell,battery)+.09*np.sum(cl[i,j])
            except Exception as exc:failures.append({'household_id':int(ids[i]),'date':str(dates[j]),'error':str(exc)})
        if (i+1)%25==0:print(f'physical {method}/{kind}: {i+1}/{H}, {time.perf_counter()-start:.1f}s',flush=True)
    assert maxnight==0
    stem=f'{method}_{kind}_seed11_test_physical';dest=ROOT/'results/dispatch';target=dest/f'{stem}.npz'
    np.savez_compressed(target,household_id=ids,date=dates,charge_kw=charge,discharge_kw=discharge,bill_aud=bills,end_energy_kwh=end,clean_day=valid,actual_load_kw=gc+cl,actual_pv_kw=pv,forecast_load_kw=load,forecast_pv_kw=forecast)
    report={'method':method,'tariff':kind,'seed':11,'n_households':H,'n_dates':D,'dates':dates.tolist(),'date_range':[str(dates[0]),str(dates[-1])],
            'clean_household_days':int(valid.sum()),'failed':len(failures),'failures':failures,'max_energy_balance_residual':maxbalance,'max_simultaneous_kw':maxsim,
            'max_terminal_error_kwh':float(np.nanmax(abs(end-5))),'max_forecast_or_scenario_night_pv_kw':maxnight,'solver_tolerance':1e-13,
            'test_used_for_fitting':False,'redesign_disclosure':REDESIGN,'night_support_in_training_validation_inference':True,
            'cost_definition':'retail settlement including fixed controlled load plus throughput proxy; no regularizer, capex or service fee',
            'mean_clean_bill_aud':float(np.mean(bills[valid])),'elapsed_seconds':time.perf_counter()-start,'output':str(target)}
    (dest/f'{stem}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='dates'}),flush=True)
    if failures:raise RuntimeError('Physical baseline has failed solves; retained, never skipped in final scoring')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['calibrated','stochastic'],required=True);ap.add_argument('--tariff',choices=['tou','flat'],required=True);main(ap.parse_args())
