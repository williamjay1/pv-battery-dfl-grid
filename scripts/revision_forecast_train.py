"""Revision-only supervised forecasters and matched decision fine-tuning.

Selection is exclusively chronological validation. Every failed candidate is
retained in its log; no test label or test metric enters training decisions.
"""
from revision_forecast_common import *
import argparse,copy
import torch
from torch import nn
from forecast_models import load_data,make_features,build_masks,set_seed
from train_decision_physical import PhysicalPVMLP
from decision_layer import BatteryLayer

class DirectMLP(nn.Module):
    def __init__(self,mode='net'):
        super().__init__();self.mode=mode
        widths=(256,128) if mode=='load' else (128,64)
        self.net=nn.Sequential(nn.Linear(774,widths[0]),nn.ReLU(),nn.Linear(widths[0],widths[1]),nn.ReLU(),nn.Linear(widths[1],48))
    def forward(self,x):
        x=self.net(x)
        return torch.nn.functional.softplus(x,beta=3) if self.mode=='load' else x

def configure():
    initialize();torch.set_num_threads(1);torch.set_num_interop_threads(1)

def initial_model(mode):
    model=PhysicalPVMLP() if mode=='pv' else DirectMLP(mode)
    path=ROOT/'checkpoints/forecast_mlp_seed11.pt' if mode=='pv' else REV/f'models/{mode}_supervised_seed11.pt'
    model.load_state_dict(torch.load(path,weights_only=False)['model_state_dict'])
    return model

def forecast_all(model,p,finite,mode):
    output=np.full(p['power_kw'].shape[:3],np.nan,np.float32);pairs=np.argwhere(finite);model.eval()
    with torch.no_grad():
        for st in range(0,len(pairs),4096):
            z=pairs[st:st+4096];h,d=z.T;pred=model(torch.from_numpy(make_features(p,z))).numpy()
            output[h,d]=pred*(p['capacity_kwp'][h,d-1,None] if mode=='pv' else 10.)
    assert np.isfinite(output[finite]).all()
    return output

def pairs_data():
    p,splits=load_data();_,finite,_,_,masks=build_masks(p,splits)
    old=json.loads((ROOT/'results/decision_training_v1.json').read_text())
    tr=np.asarray(old['train_pairs']);va=np.asarray(old['validation_pairs'])
    assert masks['train'][tr[:,0],tr[:,1]].all() and masks['validation'][va[:,0],va[:,1]].all()
    return p,finite,masks,tr,va

def supervised(args):
    configure();start=time.perf_counter();p,finite,masks,_,_=pairs_data();tr=np.argwhere(masks['train']);va=np.argwhere(masks['validation'])
    x=torch.from_numpy(make_features(p,tr));vx=torch.from_numpy(make_features(p,va))
    target=p['power_kw'][:,:,:,0]-(p['power_kw'][:,:,:,2] if args.mode=='net' else 0.)
    y=torch.from_numpy(target[tr[:,0],tr[:,1]]/10.);vy=torch.from_numpy(target[va[:,0],va[:,1]]/10.)
    configs=[{'lr':1e-3,'weight_decay':1e-4},{'lr':3e-4,'weight_decay':1e-3}];runs=[]
    for cfg in configs:
        set_seed(11);m=DirectMLP(args.mode);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);gen=torch.Generator().manual_seed(11)
        best=float('inf');state=None;epbest=0;trace=[];clock=time.perf_counter()
        for epoch in range(30):
            m.train()
            for ix in torch.randperm(len(x),generator=gen).split(512):
                loss=((m(x[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
            m.eval()
            with torch.no_grad():value=sum(float(((m(xx)-yy)**2).sum()) for xx,yy in zip(vx.split(4096),vy.split(4096)))/vy.numel()
            trace.append({'epoch':epoch+1,'validation_normalized_mse':value});print(args.mode,cfg['lr'],trace[-1],flush=True)
            if value<best-1e-8:best=value;state=copy.deepcopy(m.state_dict());epbest=epoch+1
            if epoch+1-epbest>=6:break
        m.load_state_dict(state);runs.append((m,{'config':cfg,'best_epoch':epbest,'best_validation_mse':best,'trace':trace,**runtime_info(clock)}))
    win=int(np.argmin([r[1]['best_validation_mse'] for r in runs]));m,log=runs[win]
    torch.save({'model_state_dict':m.state_dict(),'mode':args.mode,'log':log},REV/f'models/{args.mode}_supervised_seed11.pt')
    pred=forecast_all(m,p,finite,args.mode);np.savez_compressed(REV/f'forecasts/{args.mode}_supervised_seed11.npz',household_id=p['household_id'],date=p['date'],prediction_kw=pred)
    report={'status':'complete','mode':args.mode,'seed':11,'train_days':len(tr),'validation_days':len(va),'selected':win,'candidates':[r[1] for r in runs],'test_used_for_selection':False,**runtime_info(start)}
    if args.mode=='load':
        a=source_arrays();h,d=va.T;base=a['gc_shared'][h,d];truth=a['truth_gc_kw'][h,d];candidates=[]
        for w in [0,.25,.5,.75,1.]:candidates.append({'mlp_weight':w,'validation_rmse_kw':float(np.sqrt(np.mean((w*pred[h,d]+(1-w)*base-truth)**2)))})
        best=min(candidates,key=lambda r:r['validation_rmse_kw']);w=best['mlp_weight'];strong=w*pred+(1-w)*a['gc_shared'];np.savez_compressed(REV/'forecasts/load_strong_seed11.npz',household_id=p['household_id'],date=p['date'],prediction_kw=strong)
        report['deployment_ensemble_selection']={'candidates':candidates,'selected':best,'note':'If weight0 wins, report no stronger deployable load predictor was obtained; do not rename identical HGB as an improvement.'}
    write_json(REV/f'results/{args.mode}_supervised_seed11.json',report)

def selected_load(a,label):
    if label=='hgb':return a['gc_shared']
    if label=='persistence':
        out=np.full_like(a['truth_gc_kw'],np.nan);out[:,1:]=a['truth_gc_kw'][:,:-1];return out
    return np.load(REV/'forecasts/load_strong_seed11.npz')['prediction_kw']

def evaluate_model(m,x,scale,load,actual,mode,kind,battery):
    m.eval()
    with torch.no_grad():pred=m(torch.from_numpy(x)).numpy()*scale
    qp=DispatchQP(battery,kind,tolerance=1e-13);buy,sell=tariff(kind);cost=[]
    for i,row in enumerate(pred):
        c,d,_=qp.solve(load[i]-row if mode=='pv' else row);cost.append(realized_bill(actual[i],c,d,buy,sell,battery))
    return float(np.mean(cost))

def matched(args):
    if not args.pilot and all((REV/f'{folder}/{args.tag}.{suffix}').exists() for folder,suffix in [('models','pt'),('forecasts','npz'),('results','json')]):
        print('Retaining completed revision-only run:',args.tag,flush=True);return
    configure();start=time.perf_counter();p,finite,masks,tr,va=pairs_data();a=source_arrays();load=selected_load(a,args.load)
    if args.pilot:tr=tr[:256];va=va[:128]
    h,d=tr.T;vh,vd=va.T;tx=torch.from_numpy(make_features(p,tr));vx=make_features(p,va)
    scale=p['capacity_kwp'][h,d-1,None] if args.mode=='pv' else np.full((len(tr),1),10.,np.float32)
    vscale=p['capacity_kwp'][vh,vd-1,None] if args.mode=='pv' else np.full((len(va),1),10.,np.float32)
    actual=p['power_kw'][h,d,:,0]-p['power_kw'][h,d,:,2];vactual=p['power_kw'][vh,vd,:,0]-p['power_kw'][vh,vd,:,2]
    target=p['power_kw'][h,d,:,2]/scale if args.mode=='pv' else actual/10.
    ty=torch.from_numpy(target);tl=torch.from_numpy(load[h,d]);ta=torch.from_numpy(actual);ts=torch.from_numpy(scale)
    battery=Battery(throughput=args.wear);buy,sell=tariff(args.tariff);tb=torch.tensor(buy,dtype=torch.float32);ss=torch.tensor(sell,dtype=torch.float32);layer=BatteryLayer(battery,args.tariff)
    runs=[];epochs=1 if args.pilot else 4
    for lr in [1e-4,3e-5]:
        set_seed(11);m=initial_model(args.mode);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);gen=torch.Generator().manual_seed(11);clock=time.perf_counter()
        initial=evaluate_model(m,vx,vscale,load[vh,vd],vactual,args.mode,args.tariff,battery);best=initial;bestepoch=0;state=copy.deepcopy(m.state_dict());trace=[]
        for epoch in range(epochs):
            m.train();losses=[]
            for ix in torch.randperm(len(tx),generator=gen).split(32):
                out=m(tx[ix])
                if args.loss=='mse':loss=((out-ty[ix])**2).mean()
                else:
                    pred=out*ts[ix];net=tl[ix]-pred if args.mode=='pv' else pred
                    try:c,dd,e=layer(net)
                    except Exception:
                        np.savez_compressed(REV/f'results/failure_{args.tag}_lr{lr}.npz',net=net.detach().numpy(),pairs=tr[ix.numpy()]);raise
                    g=ta[ix]+c-dd;loss=(.5*(tb*torch.relu(g)+ss*torch.minimum(g,torch.zeros_like(g))+args.wear*(c+dd)).sum(-1)).mean()
                opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
            value=evaluate_model(m,vx,vscale,load[vh,vd],vactual,args.mode,args.tariff,battery);trace.append({'epoch':epoch+1,'mean_training_loss':float(np.mean(losses)),'validation_operating_excluding_cl':value});print(args.tag,lr,trace[-1],flush=True)
            if value<best-1e-6:best=value;bestepoch=epoch+1;state=copy.deepcopy(m.state_dict())
        m.load_state_dict(state);runs.append((m,{'learning_rate':lr,'start_validation_cost':initial,'best_validation_cost':best,'selected_epoch':bestepoch,'trace':trace,**runtime_info(clock)}))
    win=int(np.argmin([r[1]['best_validation_cost'] for r in runs]));m,log=runs[win]
    torch.save({'model_state_dict':m.state_dict(),'mode':args.mode,'log':log},REV/f'models/{args.tag}.pt')
    report={'status':'complete','tag':args.tag,'seed':11,'mode':args.mode,'loss':args.loss,'load':args.load,'wear':args.wear,'tariff':args.tariff,'train_days':len(tr),'validation_days':len(va),'epochs_completed':epochs,'batch_size':32,'test_used_for_selection':False,'selected':win,'candidates':[r[1] for r in runs],'process_resources':runtime_info(start)}
    if not args.pilot:
        pred=forecast_all(m,p,finite,args.mode);np.savez_compressed(REV/f'forecasts/{args.tag}.npz',household_id=p['household_id'],date=p['date'],prediction_kw=pred)
        report['full_pipeline_resources']=runtime_info(start)
    write_json(REV/f'results/{args.tag}.json',report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['supervised','matched'],required=True);ap.add_argument('--mode',choices=['pv','net','load'],default='pv');ap.add_argument('--loss',choices=['mse','dfl'],default='mse');ap.add_argument('--load',choices=['hgb','persistence','strong'],default='hgb');ap.add_argument('--wear',type=float,default=.02);ap.add_argument('--tariff',choices=['tou','flat'],default='tou');ap.add_argument('--tag');ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    if args.stage=='supervised':supervised(args)
    else:
        if not args.tag:ap.error('--tag is required')
        matched(args)
