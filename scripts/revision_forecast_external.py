"""Frozen Oct6 checkpoints on predeclared StoreNet reference scales."""
from revision_forecast_common import *
import hashlib,joblib,torch
from forecast_models import make_features,tree_features
from train_decision_physical import PhysicalPVMLP
from revision_forecast_compare import uncertainty

def main():
    initialize();torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.perf_counter();freeze=json.loads((REV/'research/external_model_freeze.json').read_text());verified=[]
    for item in freeze['artifacts']:
        path=ROOT/item['relative_path'];digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==item['sha256'],str(path);verified.append(item['relative_path'])
    hgb=joblib.load(ROOT/'checkpoints/forecast_gc_shared.joblib');outputs={}
    for label in ['2p2','2p0','2p4']:
        p=dict(np.load(REV/f'cache/storenet_external_reference_{label}kwp.npz'));mask=p['external_test_mask'];pairs=np.argwhere(mask);h,d=pairs.T;x=torch.from_numpy(make_features(p,pairs));cap=p['capacity_kwp'][h,d-1,None];load=np.maximum(hgb.predict(tree_features(p,pairs).reshape(-1,23)),0).reshape(-1,48)*10.;truth=p['power_kw'][h,d,:,0]-p['power_kw'][h,d,:,2];summary={'households':len(p['household_id']),'household_days':len(pairs),'capacity_reference_kwp':float(cap[0,0]),'fitting_on_external_data':False,'load_rmse_kw':float(np.sqrt(np.mean((load-p['power_kw'][h,d,:,0])**2))),'fees':{}};allpred={}
        for fee in ['tou','flat']:
            buy,sell=tariff(fee);qp=DispatchQP(kind=fee,tolerance=1e-13);costs={};feeout={}
            for method in ['msecont','dfl']:
                for seed in [11,23,47]:
                    path=ROOT/f'checkpoints/{"mse_continuation" if method=="msecont" else "decision"}_physical_{fee}_seed{seed}.pt';m=PhysicalPVMLP();m.load_state_dict(torch.load(path,weights_only=False)['model_state_dict']);m.eval()
                    with torch.no_grad():pred=m(x).numpy()*cap
                    allpred[f'{method}_{fee}_seed{seed}']=pred;c=np.empty_like(pred);dd=c.copy();ee=np.empty(len(pred));lat=[]
                    for i,row in enumerate(load-pred):
                        t=time.perf_counter();c[i],dd[i],e=qp.solve(row);lat.append(time.perf_counter()-t);ee[i]=e[-1]
                    cost=realized_bill(truth,c,dd,buy,sell);fullcost=np.full(mask.shape,np.nan);fullcost[mask]=cost;costs[f'{method}_seed{seed}']=fullcost
                    np.savez_compressed(REV/f'dispatch/storenet_{label}_{method}_{fee}_seed{seed}.npz',household_id=p['household_id'],date=p['date'],pairs=pairs,clean_day=mask,bill_aud=fullcost,charge_kw=c,discharge_kw=dd,forecast_load_kw=load,forecast_pv_kw=pred,actual_load_kw=p['power_kw'][h,d,:,0],actual_pv_kw=p['power_kw'][h,d,:,2],end_energy_kwh=ee)
                    feeout[f'{method}_seed{seed}']={'mean_operating_cost':float(cost.mean()),'pv_rmse_kw':float(np.sqrt(np.mean((pred-p['power_kw'][h,d,:,2])**2))),'mean_throughput_kwh':float((.5*(c+dd).sum(-1)).mean()),'median_qp_ms':float(np.median(lat)*1000)}
            differences=[]
            for seed in [11,23,47]:
                delta=costs[f'dfl_seed{seed}']-costs[f'msecont_seed{seed}'];differences.append(delta);hm=np.where(mask,delta,0).sum(1)/mask.sum(1);feeout[f'paired_seed{seed}']={'micro_mean_difference':float(delta[mask].mean()),'macro_mean_difference':float(hm.mean()),'calendar_block95':uncertainty(delta,mask,np.random.default_rng(918)),'household_differences':dict(zip(p['household_id'].astype(str),hm.tolist()))}
            mean=np.mean(differences,axis=0);feeout['three_seed_mean_paired']={'micro_mean_difference':float(mean[mask].mean()),'calendar_block95':uncertainty(mean,mask,np.random.default_rng(918))};summary['fees'][fee]=feeout;print(label,fee,feeout['three_seed_mean_paired'],flush=True)
        np.savez_compressed(REV/f'forecasts/storenet_{label}_frozen_predictions.npz',pairs=pairs,household_id=p['household_id'],date=p['date'],forecast_load_kw=load,**allpred);outputs[label]=summary;write_json(REV/'results/storenet_frozen_external.json',{'status':'running','verified_frozen_artifacts':verified,'scales':outputs,**runtime_info(start)})
    write_json(REV/'results/storenet_frozen_external.json',{'status':'complete','verified_frozen_artifacts':verified,'primary_scale':'2p2','capacity_scales_are_references_not_verified_nameplate':True,'retained_source_calendar_without_hemisphere_adjustment':True,'modeled_australian_prices_not_actual_irish_bills':True,'scales':outputs,**runtime_info(start)})

if __name__=='__main__':main()
