"""Independently tuned MSE continuation plus strictly same-LR update control."""
from pathlib import Path
import json,copy,time
import numpy as np
import torch
from forecast_models import PVMLP,load_data,make_features,build_masks,set_seed
from train_decision import evaluate
from battery_model import Battery

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p,splits=load_data();_,finite_history,_,_,masks=build_masks(p,splits)
    arr=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    training=json.loads((ROOT/'results/decision_training_v1.json').read_text())
    if training['status']!='complete':raise ValueError('DFL learning rates not final')
    tr=np.array(training['train_pairs']);va=np.array(training['validation_pairs'])
    assert np.all(masks['train'][tr[:,0],tr[:,1]]) and np.all(masks['validation'][va[:,0],va[:,1]])
    x=make_features(p,tr);vx=make_features(p,va);h,d=tr.T;vh,vd=va.T
    cap=p['capacity_kwp'][h,d-1];vcap=p['capacity_kwp'][vh,vd-1]
    target=p['power_kw'][h,d,:,2]/cap[:,None]
    vload=arr['gc_shared'][vh,vd];vactual=p['power_kw'][vh,vd,:,0]-p['power_kw'][vh,vd,:,2]
    tx=torch.from_numpy(x);ty=torch.from_numpy(target)
    pairs=np.argwhere(finite_history)
    primary={'household_id':p['household_id'],'date':p['date']}
    same_lr={'household_id':p['household_id'],'date':p['date']}
    logs=[];tuning=[];selections=[];started=time.perf_counter()

    def fit(seed,kind,cfg):
        set_seed(seed);model=PVMLP()
        model.load_state_dict(torch.load(ROOT/f'checkpoints/forecast_mlp_seed{seed}.pt',weights_only=False)['model_state_dict'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
        gen=torch.Generator().manual_seed(seed);best=evaluate(model,vx,vcap,vload,vactual,kind,Battery())
        initial=best;bestepoch=0;beststate=copy.deepcopy(model.state_dict());trace=[];clock=time.perf_counter()
        for epoch in range(cfg['epochs']):
            model.train();losses=[]
            for batch in torch.randperm(len(tx),generator=gen).split(cfg['batch_size']):
                loss=torch.mean((model(tx[batch])-ty[batch])**2)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                losses.append(float(loss.detach()))
            value=evaluate(model,vx,vcap,vload,vactual,kind,Battery())
            trace.append({'epoch':epoch+1,'train_normalized_mse':float(np.mean(losses)),'validation_bill_aud_per_day':value})
            if value<best-1e-6:best=value;bestepoch=epoch+1;beststate=copy.deepcopy(model.state_dict())
        model.load_state_dict(beststate);model.eval()
        log={'tariff':kind,'seed':seed,'config':cfg,'start_validation_bill':initial,'best_validation_bill':best,'best_epoch':bestepoch,'trace':trace,'elapsed_seconds':time.perf_counter()-clock}
        print(json.dumps(log),flush=True)
        return model,log

    def predict(model):
        result=np.full(p['power_kw'].shape[:3],np.nan,np.float32)
        with torch.no_grad():
            for i in range(0,len(pairs),4096):
                pair=pairs[i:i+4096];hh,dd=pair.T
                result[hh,dd]=model(torch.from_numpy(make_features(p,pair))).numpy()*p['capacity_kwp'][hh,dd-1,None]
        return result

    for kind in ['tou','flat']:
        cfgs=[]
        for run in training['tuning']:
            if run['tariff']==kind and run['config'] not in cfgs:cfgs.append(run['config'])
        if len(cfgs)!=2:raise ValueError(f'Expected two-LR grid for {kind}')
        candidate_models=[];candidate_logs=[]
        for cfg in cfgs:
            model,log=fit(11,kind,cfg);candidate_models.append(model);candidate_logs.append(log);tuning.append(log)
        winner=int(np.argmin([log['best_validation_bill'] for log in candidate_logs]));selected=cfgs[winner]
        dfl_cfg=next(r['config'] for r in training['final'] if r['tariff']==kind and r['seed']==11)
        strict_index=next(i for i,cfg in enumerate(cfgs) if cfg==dfl_cfg)
        selections.append({'tariff':kind,'mse_selected_config':selected,'dfl_selected_config':dfl_cfg,'same_lr_equals_primary':selected==dfl_cfg,'selection_basis':'seed11 validation bill only; same two LR candidates and epoch-zero option'})
        for seed in [11,23,47]:
            if seed==11:model,log=candidate_models[winner],candidate_logs[winner]
            else:model,log=fit(seed,kind,selected)
            pred=predict(model);primary[f'pv_msecont_{kind}_seed{seed}']=pred
            torch.save({'model_state_dict':model.state_dict(),'log':log,'role':'independently validation-selected MSE continuation'},ROOT/f'checkpoints/mse_continuation_{kind}_seed{seed}.pt')
            logs.append({**log,'role':'strong_mse_continuation'})
            if selected==dfl_cfg:strict_model,strict_log=model,log;strict_pred=pred
            else:
                if seed==11:strict_model,strict_log=candidate_models[strict_index],candidate_logs[strict_index]
                else:strict_model,strict_log=fit(seed,kind,dfl_cfg)
                strict_pred=predict(strict_model)
            same_lr[f'pv_msesamelr_{kind}_seed{seed}']=strict_pred
            torch.save({'model_state_dict':strict_model.state_dict(),'log':strict_log,'role':'DFL same-LR continuation control','identical_to_primary':selected==dfl_cfg},ROOT/f'checkpoints/mse_same_lr_{kind}_seed{seed}.pt')
            logs.append({**strict_log,'role':'strict_same_lr_control','identical_to_primary':selected==dfl_cfg})
    np.savez_compressed(ROOT/'datasets/mse_continuation_arrays_v1.npz',**primary)
    np.savez_compressed(ROOT/'datasets/mse_same_lr_arrays_v1.npz',**same_lr)
    report={'status':'complete','test_used_for_fitting_or_selection':False,'selection':'MSE continuation independently chooses its LR from the same two-candidate grid, validation bill and epoch-zero option; DFL same-LR control retained separately','same_train_and_validation_pairs_as_dfl':True,'train_household_days':len(tr),'validation_household_days':len(va),'tuning':tuning,'selections':selections,'records':logs,'elapsed_seconds':time.perf_counter()-started}
    (ROOT/'results/mse_continuation_training_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'complete','selections':selections,'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()