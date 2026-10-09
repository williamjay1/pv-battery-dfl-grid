"""Warm-start household decision learning; selection uses validation only."""
from pathlib import Path
import argparse,json,time,copy
import numpy as np
import torch
from forecast_models import PVMLP,load_data,make_features,build_masks,set_seed
from battery_model import Battery,DispatchQP,tariff,realized_bill
from decision_layer import BatteryLayer

ROOT=Path(__file__).resolve().parents[1]

def evaluate(model,x,cap,load,actual,kind,battery):
    model.eval()
    with torch.no_grad(): pred=model(torch.from_numpy(x)).numpy()*cap[:,None]
    qp=DispatchQP(battery,kind,tolerance=1e-13);buy,sell=tariff(kind)
    bills=[]
    for i in range(len(pred)):
        c,d,_=qp.solve(load[i]-pred[i]);bills.append(realized_bill(actual[i],c,d,buy,sell,battery))
    return float(np.mean(bills))

def train_one(p,arr,trainpairs,valpairs,seed,kind,cfg,battery):
    set_seed(seed);model=PVMLP()
    state=torch.load(ROOT/f'checkpoints/forecast_mlp_seed{seed}.pt',weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    x=make_features(p,trainpairs);vx=make_features(p,valpairs)
    h,d=trainpairs.T;vh,vd=valpairs.T
    cap=p['capacity_kwp'][h,d-1];vcap=p['capacity_kwp'][vh,vd-1]
    load=arr['gc_shared'][h,d];vload=arr['gc_shared'][vh,vd]
    actual=p['power_kw'][h,d,:,0]-p['power_kw'][h,d,:,2]
    vactual=p['power_kw'][vh,vd,:,0]-p['power_kw'][vh,vd,:,2]
    layer=BatteryLayer(battery,kind);buy,sell=tariff(kind)
    tbuy=torch.tensor(buy,dtype=torch.float32);tsell=torch.tensor(sell,dtype=torch.float32)
    tx=torch.from_numpy(x);tc=torch.from_numpy(cap);tl=torch.from_numpy(load);ta=torch.from_numpy(actual)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    start=time.perf_counter();baseline=evaluate(model,vx,vcap,vload,vactual,kind,battery)
    best=baseline;beststate=copy.deepcopy(model.state_dict());bestepoch=0;trace=[]
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(cfg['epochs']):
        model.train();perm=torch.randperm(len(tx),generator=generator);losses=[]
        for batch in perm.split(cfg['batch_size']):
            pred=model(tx[batch])*tc[batch,None]
            net_input=tl[batch]-pred
            try:c,d,e=layer(net_input)
            except Exception:
                np.savez_compressed(ROOT/f'results/decision_failure_{kind}_{seed}.npz',net=net_input.detach().numpy(),pairs=trainpairs[batch.numpy()],actual=ta[batch].numpy())
                raise
            g=ta[batch]+c-d
            cost=battery.dt*(tbuy*torch.relu(g)+tsell*torch.minimum(g,torch.zeros_like(g))+battery.throughput*(c+d)).sum(-1)
            loss=cost.mean();opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite network gradient')
            opt.step();losses.append(float(loss.detach()))
        val=evaluate(model,vx,vcap,vload,vactual,kind,battery)
        trace.append({'epoch':epoch+1,'train_bill_aud_per_day':float(np.mean(losses)),'validation_bill_aud_per_day':val,'elapsed_seconds':time.perf_counter()-start})
        print(json.dumps({'seed':seed,'tariff':kind,'lr':cfg['learning_rate'],**trace[-1]}),flush=True)
        if val<best-1e-6:best=val;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch+1
    model.load_state_dict(beststate)
    log={'seed':seed,'tariff':kind,'config':cfg,'validation_start_bill':baseline,'validation_best_bill':best,'best_epoch':bestepoch,'trace':trace,'training_household_days':len(trainpairs),'validation_household_days':len(valpairs),'runtime_seconds':time.perf_counter()-start}
    return model,log

def run(args):
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    p,splits=load_data();_,finite_history,timecode,role,masks=build_masks(p,splits)
    arr=np.load(ROOT/'datasets/forecast_arrays_v1.npz');train=np.argwhere(masks['train']);val=np.argwhere(masks['validation'])
    rng=np.random.default_rng(3107)
    trainpairs=train[np.sort(rng.choice(len(train),min(args.train_days,len(train)),replace=False))]
    valpairs=val[np.sort(rng.choice(len(val),min(args.val_days,len(val)),replace=False))]
    battery=Battery()
    report={'status':'running','training_source':'known households, training dates only; fixed uniform subsample across household-days; no test-based selection','validation_source':'fixed random validation subset, same across seeds/configs','train_pairs':trainpairs.tolist(),'validation_pairs':valpairs.tolist(),'battery':battery.__dict__,'tuning':[],'final':[],'test_used_for_selection':False}
    models={}
    for kind in args.tariffs:
        runs=[]
        for lr in args.learning_rates:
            cfg={'learning_rate':lr,'epochs':args.epochs,'batch_size':args.batch_size}
            model,log=train_one(p,arr,trainpairs,valpairs,11,kind,cfg,battery);runs.append((model,log));report['tuning'].append(log)
            (ROOT/f'results/decision_training_{args.tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        chosen=int(np.argmin([r[1]['validation_best_bill'] for r in runs]))
        for seed in args.seeds:
            if seed==11:model,log=runs[chosen]
            else:model,log=train_one(p,arr,trainpairs,valpairs,seed,kind,runs[chosen][1]['config'],battery)
            model.eval();models[(kind,seed)]=model;report['final'].append(log)
            torch.save({'model_state_dict':model.state_dict(),'seed':seed,'tariff':kind,'log':log,'battery':battery.__dict__},ROOT/f'checkpoints/decision_{args.tag}_{kind}_seed{seed}.pt')
    pairs=np.argwhere(finite_history);out={'household_id':p['household_id'],'date':p['date']}
    for (kind,seed),model in models.items():
        result=np.full(p['power_kw'].shape[:3],np.nan,np.float32)
        with torch.no_grad():
            for i in range(0,len(pairs),4096):
                pair=pairs[i:i+4096];h,d=pair.T
                pred=model(torch.from_numpy(make_features(p,pair))).numpy()*p['capacity_kwp'][h,d-1,None]
                result[h,d]=pred
        out[f'pv_dfl_{kind}_seed{seed}']=result
    np.savez_compressed(ROOT/f'datasets/decision_arrays_{args.tag}.npz',**out)
    report['status']='complete';(ROOT/f'results/decision_training_{args.tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE '+args.tag,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v1');ap.add_argument('--train-days',type=int,default=4096);ap.add_argument('--val-days',type=int,default=1024);ap.add_argument('--epochs',type=int,default=4);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--learning-rates',type=float,nargs='+',default=[1e-4,3e-5]);ap.add_argument('--seeds',type=int,nargs='+',default=[11,23,47]);ap.add_argument('--tariffs',nargs='+',default=['tou','flat']);ap.add_argument('--threads',type=int,default=2)
    run(ap.parse_args())
