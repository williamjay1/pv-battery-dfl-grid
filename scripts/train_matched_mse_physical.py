"""Same-budget independently tuned MSE continuation under fixed night support."""
from pathlib import Path
import argparse,copy,json,time
import numpy as np
import torch
from forecast_models import make_features,set_seed
from train_decision import evaluate
from battery_model import Battery
from train_decision_physical import ROOT,PhysicalPVMLP,NIGHT_SLOTS,REDESIGN,frozen_inputs,predict_all,merge

def main(args):
    if args.merge:merge('mse_continuation');return
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    kind=args.tariff;p,finite,_,tr,va=frozen_inputs();arr=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    physical=json.loads((ROOT/f'results/decision_training_physical_{kind}.json').read_text())
    assert physical['status']=='complete'
    assert np.array_equal(tr,physical['train_pairs']) and np.array_equal(va,physical['validation_pairs'])
    h,d=tr.T;vh,vd=va.T;cap=p['capacity_kwp'][h,d-1];vcap=p['capacity_kwp'][vh,vd-1]
    tx=torch.from_numpy(make_features(p,tr));ty=torch.from_numpy(p['power_kw'][h,d,:,2]/cap[:,None]);vx=make_features(p,va)
    vload=arr['gc_shared'][vh,vd];vactual=p['power_kw'][vh,vd,:,0]-p['power_kw'][vh,vd,:,2]
    configs=[r['config'] for r in physical['tuning']];assert len(configs)==2
    started=time.perf_counter()
    def fit(seed,cfg):
        set_seed(seed);model=PhysicalPVMLP();model.load_state_dict(torch.load(ROOT/f'checkpoints/forecast_mlp_seed{seed}.pt',weights_only=False)['model_state_dict'])
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4);gen=torch.Generator().manual_seed(seed)
        best=evaluate(model,vx,vcap,vload,vactual,kind,Battery());initial=best;state=copy.deepcopy(model.state_dict());epochbest=0;trace=[];clock=time.perf_counter()
        for epoch in range(cfg['epochs']):
            model.train();losses=[]
            for ix in torch.randperm(len(tx),generator=gen).split(cfg['batch_size']):
                loss=((model(tx[ix])-ty[ix])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            value=evaluate(model,vx,vcap,vload,vactual,kind,Battery());trace.append({'epoch':epoch+1,'train_normalized_mse':float(np.mean(losses)),'validation_bill_aud_per_day':value})
            if value<best-1e-6:best=value;state=copy.deepcopy(model.state_dict());epochbest=epoch+1
        model.load_state_dict(state);model.eval();log={'tariff':kind,'seed':seed,'config':cfg,'start_validation_bill':initial,'best_validation_bill':best,'best_epoch':epochbest,'trace':trace,'elapsed_seconds':time.perf_counter()-clock}
        print(json.dumps(log),flush=True);return model,log
    candidates=[fit(11,cfg) for cfg in configs];win=int(np.argmin([r[1]['best_validation_bill'] for r in candidates]));chosen=configs[win]
    dfl_cfg=next(r['config'] for r in physical['final'] if r['seed']==11);strict_index=next(i for i,cfg in enumerate(configs) if cfg==dfl_cfg)
    primary={'household_id':p['household_id'],'date':p['date']};strict={'household_id':p['household_id'],'date':p['date']};records=[]
    for seed in [11,23,47]:
        model,log=candidates[win] if seed==11 else fit(seed,chosen);pred=predict_all(model,p,finite);primary[f'pv_msecont_{kind}_seed{seed}']=pred
        torch.save({'model_state_dict':model.state_dict(),'model_class':'PhysicalPVMLP','night_zero_slots':NIGHT_SLOTS,'log':log,'role':'independently tuned physical MSE continuation'},ROOT/f'checkpoints/mse_continuation_physical_{kind}_seed{seed}.pt');records.append({**log,'role':'primary'})
        if chosen==dfl_cfg:sm,sl,sp=model,log,pred
        else:
            sm,sl=candidates[strict_index] if seed==11 else fit(seed,dfl_cfg);sp=predict_all(sm,p,finite)
        strict[f'pv_msesamelr_{kind}_seed{seed}']=sp
        torch.save({'model_state_dict':sm.state_dict(),'model_class':'PhysicalPVMLP','night_zero_slots':NIGHT_SLOTS,'log':sl,'role':'same-LR physical control','identical_to_primary':chosen==dfl_cfg},ROOT/f'checkpoints/mse_same_lr_physical_{kind}_seed{seed}.pt');records.append({**sl,'role':'strict_same_lr','identical_to_primary':chosen==dfl_cfg})
    np.savez_compressed(ROOT/f'datasets/mse_continuation_arrays_physical_{kind}.npz',**primary)
    np.savez_compressed(ROOT/f'datasets/mse_same_lr_arrays_physical_{kind}.npz',**strict)
    report={'status':'complete','tariff':kind,'redesign_disclosure':REDESIGN,'model_class':'PhysicalPVMLP','night_zero_slots':NIGHT_SLOTS,'same_train_and_validation_pairs_as_original_and_physical_dfl':True,'train_household_days':len(tr),'validation_household_days':len(va),'test_used_for_fitting_or_selection':False,'tuning':[r[1] for r in candidates],'selected_config':chosen,'dfl_selected_config':dfl_cfg,'same_lr_equals_primary':chosen==dfl_cfg,'records':records,'elapsed_seconds':time.perf_counter()-started}
    (ROOT/f'results/mse_continuation_training_physical_{kind}.json').write_text(json.dumps(report,indent=2));print('COMPLETE physical MSE '+kind,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tariff',choices=['tou','flat']);ap.add_argument('--merge',action='store_true');args=ap.parse_args()
    if not args.merge and args.tariff is None:ap.error('--tariff required unless --merge')
    main(args)
