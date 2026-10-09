"""Post-freeze accuracy diagnostics, not a further model selection stage."""
from revision_forecast_common import *
from revision_forecast_train import selected_load

def main():
    a=source_arrays();out={'selection':'All model/ensemble choices were frozen on validation before these test metrics. These diagnostics do not trigger refitting.','metrics':{}};load={k:selected_load(a,k) for k in ['hgb','persistence','strong']};net=np.load(REV/'forecasts/net_supervised_seed11.npz')['prediction_kw'];truthnet=a['truth_gc_kw']-a['truth_pv_kw'];pv={k:old_signal(a,k,'tou') for k in ['msecont','dfl']}
    for label,role in [('known244',0),('heldout55',1)]:
        mask=(a['household_role']==role)[:,None]&(a['split_code']==2)[None,:]&a['valid_history']&a['valid_target_gcgg'];z={'household_days':int(mask.sum()),'load':{},'net':{},'pv':{}}
        def score(pred,truth):
            e=(pred-truth)[mask];assert np.isfinite(e).all();return {'rmse_kw':float(np.sqrt(np.mean(e**2))),'mae_kw':float(np.mean(abs(e))),'bias_kw':float(e.mean())}
        for name,p in load.items():z['load'][name]=score(p,a['truth_gc_kw'])
        z['net']['direct_net_mse']=score(net,truthnet)
        for name,p in pv.items():z['pv'][name]=score(p,a['truth_pv_kw']);z['net']['hgb_minus_'+name]=score(load['hgb']-p,truthnet)
        out['metrics'][label]=z
    write_json(REV/'results/revision_predictive_metrics.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
