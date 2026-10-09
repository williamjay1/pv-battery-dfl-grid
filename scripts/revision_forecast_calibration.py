"""Training-only small cost calibration; chronological validation shrinkage."""
from revision_forecast_common import *
import argparse
import torch
from decision_layer import BatteryLayer
from decision_baselines import prepare,BANDS

def design(dates,mode):
    months=np.asarray([int(str(d)[5:7]) for d in dates]);season=(months%12)//3
    bands=BANDS if mode in ('seasonal12','calibrated3') else np.eye(6).repeat(8,axis=1)
    if mode=='calibrated3':return np.broadcast_to(bands,(len(dates),3,48)).copy()
    return np.einsum('ds,kt->dskt',np.eye(4)[season],bands).reshape(len(dates),-1,48)

def main(args):
    initialize();torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.perf_counter();a=source_arrays()
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text());_,_,train,val=prepare(a,splits)
    isnet=args.mode.startswith('net_calibrated24');nightnet=args.mode.endswith('_night');base=np.load(REV/'forecasts/net_supervised_seed11.npz')['prediction_kw'] if isnet else a['pv_mlp_seed11']
    features=design(a['date'],args.mode);nparam=features.shape[1]
    th,td=train.T;vh,vd=val.T;cap=a['capacity_kwp'][:,np.maximum(np.arange(len(a['date']))-1,0)]
    tb=torch.tensor(base[th,td],dtype=torch.float64);tc=torch.tensor(cap[th,td,None],dtype=torch.float64);tf=torch.tensor(features[td],dtype=torch.float64)
    tl=torch.tensor(a['gc_shared'][th,td],dtype=torch.float64);actual=torch.tensor(a['truth_gc_kw'][th,td]-a['truth_pv_kw'][th,td],dtype=torch.float64)
    torch.manual_seed(7814);raw=torch.nn.Parameter(torch.zeros(nparam,dtype=torch.float64));opt=torch.optim.Adam([raw],lr=.03);layer=BatteryLayer(kind=args.tariff);buy,sell=tariff(args.tariff);buyt=torch.tensor(buy);sellt=torch.tensor(sell);rng=np.random.default_rng(7815);trace=[];nightmask=torch.tensor(np.isin(np.arange(48),NIGHT))[None,:]
    for epoch in range(3):
        losses=[]
        for ix in np.array_split(rng.permutation(len(train)),len(train)//32):
            shift=torch.einsum('k,bkt->bt',.25*torch.tanh(raw),tf[ix]);pred=tb[ix]+tc[ix]*shift
            if not isnet:pred=torch.clamp(pred,min=0)
            if nightnet:pred=torch.where(nightmask,torch.clamp(pred,min=0),pred)
            net=pred if isnet else tl[ix]-pred;c,d,e=layer(net);g=actual[ix]+c-d
            loss=(.5*(buyt*torch.relu(g)+sellt*torch.minimum(g,torch.zeros_like(g))+.02*(c+d)).sum(-1)).mean()
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        trace.append({'epoch':epoch+1,'mean_training_operating_excluding_cl':float(np.mean(losses)),'bias_normalized':(.25*torch.tanh(raw)).detach().tolist()});print(args.mode,args.tariff,trace[-1],flush=True)
    beta=(.25*torch.tanh(raw)).detach().numpy();qp=DispatchQP(kind=args.tariff,tolerance=1e-13);candidates=[]
    for alpha in [0,.25,.5,.75,1.]:
        pred=base[vh,vd]+cap[vh,vd,None]*np.einsum('k,bkt->bt',alpha*beta,features[vd]);cost=[]
        if not isnet:pred=np.maximum(pred,0)
        if nightnet:pred[:,NIGHT]=np.maximum(pred[:,NIGHT],0)
        for i,(h,d) in enumerate(val):
            c,dd,_=qp.solve(pred[i] if isnet else a['gc_shared'][h,d]-pred[i]);cost.append(realized_bill(a['truth_gc_kw'][h,d]-a['truth_pv_kw'][h,d],c,dd,buy,sell))
        candidates.append({'alpha':alpha,'validation_operating_excluding_cl':float(np.mean(cost))})
    best=min(candidates,key=lambda v:v['validation_operating_excluding_cl']);coef=best['alpha']*beta
    pred=base+cap[:,:,None]*np.einsum('k,dkt->dt',coef,features)[None,:,:]
    if not isnet:pred=np.maximum(pred,0)
    if nightnet:pred[:,:,NIGHT]=np.maximum(pred[:,:,NIGHT],0)
    tag=f'{args.mode}_{args.tariff}_seed11';np.savez_compressed(REV/f'forecasts/{tag}.npz',household_id=a['household_id'],date=a['date'],prediction_kw=pred.astype(np.float32))
    write_json(REV/f'results/{tag}.json',{'status':'complete','method':args.mode,'n_parameters':nparam,'tariff':args.tariff,'seed':11,'train_days':len(train),'validation_days':len(val),'training_pairs':train.tolist(),'validation_pairs':val.tolist(),'daylight_bands':['06-10','10-14','14-18'] if not isnet else None,'net_time_bands':'six fixed 4-hour blocks' if isnet else None,'seasons':'DJF,MAM,JJA,SON','max_absolute_offset_per_capacity':.25,'selected':best,'coefficients':coef.tolist(),'trace':trace,'validation_candidates':candidates,'night_nonnegative_net_projection_in_calibration_training_validation_inference':nightnet,'redesign_disclosure':'Night projection added after this revision diagnosed negative night direct-net signals; old signed results retained. Foundation net MLP is not retrained. No new independent confirmation.' if nightnet else None,'test_used_for_selection':False,'training_resources':runtime_info(start)})
    if args.dispatch:dispatch(a,tag+'_expanded',args.tariff,net_signal=pred if isnet else None,pv_signal=None if isnet else pred)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['seasonal12','net_calibrated24','net_calibrated24_night','calibrated3'],required=True);ap.add_argument('--tariff',choices=['tou','flat'],default='tou');ap.add_argument('--dispatch',action='store_true');main(ap.parse_args())
