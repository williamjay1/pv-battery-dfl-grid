"""Leakage-controlled day-ahead PV/load baselines for the Ausgrid experiment.

Reusable API: load_data(), make_features(panel, pairs), PVMLP, build_masks().
pairs[:,0] indexes household and pairs[:,1] indexes target source-clock date.
All features precede the target day. Fixed physical normalization is not fitted.
"""
from pathlib import Path
import argparse, json, os, time, random
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[11,23,47]
INPUT_DIM=774

class PVMLP(nn.Module):
    def __init__(self,input_dim=INPUT_DIM):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,48))
    def forward(self,x):
        # Nonnegative normalized PV, without clipping predictions to observed maxima.
        return torch.nn.functional.softplus(self.net(x),beta=3.0)

def load_data(root=ROOT):
    p=np.load(root/'datasets/ausgrid_source_clock_panel.npz')
    panel={k:p[k] for k in p.files}
    splits=json.loads((root/'datasets/splits_v1.json').read_text(encoding='utf-8'))
    return panel,splits

def build_masks(p,splits):
    H,D=p['valid_gc_gg_day'].shape
    history=np.zeros((H,D),bool); finite_history=np.zeros((H,D),bool)
    finite=np.isfinite(p['power_kw'][:,:,:,[0,2]]).all(axis=(2,3))
    for d in range(7,D):
        history[:,d]=p['valid_gc_gg_day'][:,d-7:d].all(1)
        finite_history[:,d]=finite[:,d-7:d].all(1)
    dates=p['date']; timecode=np.select([dates<='2011-12-31',dates<='2012-06-30'],[0,1],default=2).astype(np.int8)
    role=np.full(H,2,np.int8);role[np.isin(p['household_id'],splits['known_household_ids'])]=0;role[np.isin(p['household_id'],splits['heldout_household_ids'])]=1
    clean=history&p['valid_gc_gg_day']
    masks={name:clean&(role==0)[:,None]&(timecode==v)[None,:] for name,v in [('train',0),('validation',1),('test',2)]}
    masks['heldout_test']=clean&(role==1)[:,None]&(timecode==2)[None,:]
    return history,finite_history,timecode,role,masks

def calendar_features(dates):
    dt=np.asarray(dates,dtype='datetime64[D]')
    day=(dt-dt.astype('datetime64[Y]')).astype(int)
    weekday=(dt.astype(int)+3)%7 # Monday=0, 1970-01-01 Thursday
    return np.stack([np.sin(2*np.pi*day/365.25),np.cos(2*np.pi*day/365.25),np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),(weekday>=5).astype(float)],axis=1).astype(np.float32)

def make_features(p,pairs):
    pairs=np.asarray(pairs,dtype=int); h,d=pairs.T
    cap=p['capacity_kwp'][h,d-1]
    hist=p['power_kw'][h[:,None],d[:,None]+np.arange(-7,0)[None,:]]
    gc=hist[:,:,:,0].reshape(len(pairs),336)/10.0
    pv=hist[:,:,:,2].reshape(len(pairs),336)/cap[:,None]
    cal=calendar_features(p['date'][d])
    x=np.concatenate([gc,pv,gc[:,-48:],pv[:,-48:],cal,cap[:,None]/10.0],axis=1).astype(np.float32)
    if x.shape[1]!=INPUT_DIM or not np.isfinite(x).all(): raise ValueError('Invalid historical feature matrix')
    return x

def metrics(y,pred,mask,cap):
    yy=y[mask]; pp=pred[mask]; cc=np.broadcast_to(cap[:,:,None],y.shape)[mask]
    good=np.isfinite(yy)&np.isfinite(pp)&np.isfinite(cc)
    if not good.all(): raise ValueError('Nonfinite scored target or prediction')
    e=pp-yy
    return {'household_days':int(mask.sum()),'mae_kw':float(np.abs(e).mean()),'rmse_kw':float(np.sqrt((e*e).mean())), 'capacity_normalized_mae':float((np.abs(e)/cc).mean()),'capacity_normalized_rmse':float(np.sqrt(((e/cc)**2).mean())), 'bias_kw':float(e.mean())}

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def train_mlp(xtr,ytr,xv,yv,seed,config,outpath):
    set_seed(seed); model=PVMLP();opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tx=torch.from_numpy(xtr);ty=torch.from_numpy(ytr);vx=torch.from_numpy(xv);vy=torch.from_numpy(yv)
    best=float('inf');best_state=None;best_epoch=-1;trace=[];start=time.perf_counter()
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(config['max_epochs']):
        model.train();perm=torch.randperm(len(tx),generator=generator); total=0
        for batch in perm.split(config['batch_size']):
            pred=model(tx[batch]);loss=((pred-ty[batch])**2).mean();opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad():
            val=sum(float(((model(a)-b)**2).sum()) for a,b in zip(vx.split(4096),vy.split(4096)))/vy.numel()
        trace.append({'epoch':epoch+1,'train_mse':total/len(tx),'validation_mse':val,'elapsed_seconds':time.perf_counter()-start})
        if val<best-1e-8:
            best=val;best_epoch=epoch;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch>=config['patience']: break
    model.load_state_dict(best_state)
    torch.save({'model_state_dict':best_state,'input_dim':INPUT_DIM,'seed':seed,'config':config,'best_epoch':best_epoch+1,'validation_normalized_mse':best,'feature_contract':'past7GC/10,past7PV/cap,past1GC/10,past1PV/cap,annual_sin_cos,weekday_sin_cos,weekend,cap/10','training_pairs_clean':len(xtr)},outpath)
    return model,{'seed':seed,'config':config,'best_epoch':best_epoch+1,'validation_normalized_mse':best,'runtime_seconds':time.perf_counter()-start,'trace':trace,'checkpoint':str(outpath)}

def tree_features(p,pairs):
    """Fixed compact hour-conditional features, returning [day,48,feature]."""
    h,d=np.asarray(pairs).T; cap=p['capacity_kwp'][h,d-1]
    hist=p['power_kw'][h[:,None],d[:,None]+np.arange(-7,0)[None,:]]
    pv=hist[:,:,:,2]/cap[:,None,None];gc=hist[:,:,:,0]/10
    slot=np.arange(48);cal=calendar_features(p['date'][d])
    pieces=[pv[:,-1],pv[:,0],pv.mean(1),pv.std(1),np.max(pv,1),np.min(pv,1),gc[:,-1],gc[:,0],gc.mean(1),gc.std(1)]
    scalar=[pv[:,-1].mean(1),pv[:,-1].max(1),gc[:,-1].mean(1),gc[:,-1].max(1),cap/10]+[cal[:,j] for j in range(cal.shape[1])]
    pieces.extend([np.broadcast_to(a[:,None],(len(pairs),48)) for a in scalar])
    pieces.extend([np.broadcast_to(a[None,:],(len(pairs),48)) for a in [np.sin(2*np.pi*slot/48),np.cos(2*np.pi*slot/48),slot/47]])
    return np.stack(pieces,axis=-1).astype(np.float32)

def run(root=ROOT,threads=4,max_epochs=30):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import joblib
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    start=time.perf_counter();p,splits=load_data(root);history,finite_history,timecode,role,masks=build_masks(p,splits)
    for directory in ['results','checkpoints']:(root/directory).mkdir(exist_ok=True)
    train=np.argwhere(masks['train']);val=np.argwhere(masks['validation']);allpairs=np.argwhere(finite_history)
    cap=p['capacity_kwp'];truth=p['power_kw']
    xtr=make_features(p,train);xv=make_features(p,val)
    ytr=truth[train[:,0],train[:,1],:,2]/cap[train[:,0],train[:,1],None]
    yv=truth[val[:,0],val[:,1],:,2]/cap[val[:,0],val[:,1],None]
    report={'status':'running','selection':'validation only; test metrics computed after frozen fitting','threads':threads,'n_train':len(train),'n_validation':len(val),'tuning':[],'seeds':SEEDS,'hardware':'CPU only','failures':[]}
    arr={'household_id':p['household_id'],'date':p['date'],'truth_gc_kw':truth[:,:,:,0],'truth_cl_kw':truth[:,:,:,1],'truth_pv_kw':truth[:,:,:,2], 'capacity_kwp':cap,'valid_target_gcgg':p['valid_gc_gg_day'],'valid_target_feeder':p['valid_feeder_day'],'valid_history':history,'valid_input_finite':finite_history,'split_code':timecode,'household_role':role}
    for name,lag in [('pv_persistence',1),('pv_weekly',7)]:
        dest=np.full(truth.shape[:3],np.nan,np.float32);dest[:,lag:]=truth[:,:-lag,:,2];arr[name]=dest
    configs=[{'learning_rate':1e-3,'weight_decay':1e-4,'batch_size':512,'max_epochs':max_epochs,'patience':6},{'learning_rate':5e-4,'weight_decay':1e-3,'batch_size':512,'max_epochs':max_epochs,'patience':6}]
    models=[]
    for i,cfg in enumerate(configs):
        print(f'Tuning MLP {i+1}/{len(configs)}',flush=True)
        model,runlog=train_mlp(xtr,ytr,xv,yv,11,cfg,root/f'checkpoints/forecast_mlp_tune{i}_seed11.pt');report['tuning'].append(runlog);models.append(model)
    selected=int(np.argmin([t['validation_normalized_mse'] for t in report['tuning']]))
    report['selected_configuration_index']=selected;report['final_training']=[]
    for seed in SEEDS:
        if seed==11:
            model=models[selected];log=report['tuning'][selected].copy()
            torch.save(torch.load(log['checkpoint'],weights_only=False),root/f'checkpoints/forecast_mlp_seed{seed}.pt');log['checkpoint']=str(root/f'checkpoints/forecast_mlp_seed{seed}.pt')
        else:
            print(f'Final MLP seed {seed}',flush=True)
            model,log=train_mlp(xtr,ytr,xv,yv,seed,configs[selected],root/f'checkpoints/forecast_mlp_seed{seed}.pt')
        report['final_training'].append(log)
        output=np.full(truth.shape[:3],np.nan,np.float32);model.eval()
        with torch.no_grad():
            for i in range(0,len(allpairs),4096):
                pair=allpairs[i:i+4096];x=make_features(p,pair);pred=model(torch.from_numpy(x)).numpy()*cap[pair[:,0],pair[:,1],None];output[pair[:,0],pair[:,1]]=pred
        arr[f'pv_mlp_seed{seed}']=output
        (root/'results/forecast_training_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Pooled trees receive the same historical information compressed into fixed summaries.
    # A fixed random household-day subset bounds CPU/memory; no test or future labels are used.
    rng=np.random.default_rng(1907);selected_days=np.sort(rng.choice(len(train),size=min(12500,len(train)),replace=False));treepairs=train[selected_days]
    xt=tree_features(p,treepairs).reshape(-1,23)
    # All 48 horizons retained for each selected household-day; 600,000 samples at most.
    vt_idx=np.linspace(0,len(val)-1,min(2500,len(val)),dtype=int);vpair=val[vt_idx];vxt=tree_features(p,vpair).reshape(-1,23)
    report['tree_training']={};
    for name,ch,scale in [('pv_hgb',2,'capacity'),('gc_shared',0,'fixed10')]:
        den=cap[treepairs[:,0],treepairs[:,1],None] if ch==2 else 10.0
        yt=(truth[treepairs[:,0],treepairs[:,1],:,ch]/den).reshape(-1)
        vd=cap[vpair[:,0],vpair[:,1],None] if ch==2 else 10.0;vyt=(truth[vpair[:,0],vpair[:,1],:,ch]/vd).reshape(-1)
        candidates=[];treelog=[]
        for leaves in [15,31]:
            t0=time.perf_counter();print(f'Training {name} max_leaf_nodes={leaves}',flush=True)
            model=HistGradientBoostingRegressor(max_iter=140,learning_rate=.07,max_leaf_nodes=leaves,l2_regularization=1.0,min_samples_leaf=40,early_stopping=False,random_state=11)
            model.fit(xt,yt);loss=float(np.mean((np.maximum(model.predict(vxt),0)-vyt)**2));candidates.append(model);treelog.append({'max_leaf_nodes':leaves,'validation_mse':loss,'runtime_seconds':time.perf_counter()-t0})
        best=int(np.argmin([e['validation_mse'] for e in treelog]));model=candidates[best];joblib.dump(model,root/f'checkpoints/forecast_{name}.joblib')
        report['tree_training'][name]={'sampled_training_household_days':len(treepairs),'training_half_hours':len(yt),'validation_household_days':len(vpair),'tuning':treelog,'selected_index':best}
        output=np.full(truth.shape[:3],np.nan,np.float32)
        for i in range(0,len(allpairs),2048):
            pair=allpairs[i:i+2048];features=tree_features(p,pair);pred=np.maximum(model.predict(features.reshape(-1,23)),0).reshape(-1,48)
            pred*=cap[pair[:,0],pair[:,1],None] if ch==2 else 10.0;output[pair[:,0],pair[:,1]]=pred
        arr[name]=output
    report['metrics']={}
    for key in ['pv_persistence','pv_weekly','pv_hgb']+[f'pv_mlp_seed{s}' for s in SEEDS]:
        report['metrics'][key]={name:metrics(truth[:,:,:,2],arr[key],mask,cap) for name,mask in masks.items() if name!='train'}
    report['load_metrics']={name:metrics(truth[:,:,:,0],arr['gc_shared'],mask,cap) for name,mask in masks.items() if name!='train'}
    report['load_metric_note']='Load normalized metrics use PV capacity only for a common scale; interpret primary load MAE/RMSE in kW.'
    report['software_versions']={'numpy':np.__version__,'torch':torch.__version__}
    import sklearn;report['software_versions']['sklearn']=sklearn.__version__
    report['runtime_seconds']=time.perf_counter()-start;report['status']='complete'
    target=root/'datasets/forecast_arrays_v1.npz';print('Saving full forecast arrays',flush=True);np.savez_compressed(target,**arr)
    report['array_file']=str(target);report['array_bytes']=target.stat().st_size
    (root/'results/forecast_training_v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'runtime_seconds':report['runtime_seconds'],'metrics':report['metrics'],'load_metrics':report['load_metrics']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--max-epochs',type=int,default=30);args=ap.parse_args();run(args.root,args.threads,args.max_epochs)
