"""Revision-only continuous-SOC terminal-value control and matched learning.

The common daily terminal equality is replaced by a dummy fixed variable,
preserving the audited KKT routine's full-rank equality-count interface.
All objectives retain the original normalized strictly convex tie-break.
"""
from revision_forecast_common import *
import argparse,copy
import torch
import clarabel
from scipy import sparse,linalg
from decision_layer import polished_solution
from revision_forecast_train import configure,pairs_data,initial_model,forecast_all
from forecast_models import make_features,set_seed

RHO=.01
class TerminalQP(DispatchQP):
    def __init__(self,battery=Battery(),kind='tou'):
        super().__init__(battery,kind,tolerance=1e-13);n=self.n
        self.value_lambda=(float(self.buy.min())+battery.throughput)/battery.efficiency
        aa=sparse.hstack([self.A,sparse.csc_matrix((self.A.shape[0],1))],format='lil');aa[n,:]=0.;aa[n,-1]=1.;self.A=aa.tocsc();self.rhs[n]=0.
        pp=sparse.block_diag([self.P,sparse.csc_matrix((1,1))],format='lil');pp[3*n-1,3*n-1]+=2*RHO;self.P=pp.tocsc();self.q=np.r_[self.q,0.];self.q[3*n-1]+=-10*RHO-self.value_lambda;self.dim+=1
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_gap_abs=1e-13;settings.tol_gap_rel=1e-13;settings.tol_feas=1e-13;settings.max_iter=200
        self.solver=clarabel.DefaultSolver(self.P,self.q,self.A,self.rhs,[clarabel.ZeroConeT(n+1),clarabel.NonnegativeConeT(self.A.shape[0]-n-1)],settings)
    def state(self,soc):
        self.rhs[0]=float(soc);return self
    def value(self,e):return self.value_lambda*e-RHO*(e-5)**2

class _Terminal(torch.autograd.Function):
    @staticmethod
    def forward(ctx,net,soc,kind):
        rows=net.detach().double().numpy();initial=soc.detach().double().numpy();qp=TerminalQP(kind=kind);outputs=[];states=[]
        for row,e0 in zip(rows,initial):
            c,d,e,state,_=polished_solution(qp.state(e0),row);outputs.append(np.stack([c,d,e]));states.append(state)
        ctx.states=states;ctx.dtype=net.dtype
        return torch.as_tensor(np.stack(outputs),dtype=net.dtype)
    @staticmethod
    def backward(ctx,grad):
        gn=[];gs=[]
        for g,state in zip(grad.detach().double().numpy(),ctx.states):
            kkt,ix,rows,dim=state;gx=np.r_[g.ravel(),np.zeros(dim-144)];adj=linalg.solve(kkt,np.r_[gx,np.zeros(len(ix))],assume_a='sym',check_finite=False)[dim:];db=np.zeros(rows);db[ix]=adj;gn.append(-db[49:97]);gs.append(db[0])
        return torch.as_tensor(np.stack(gn),dtype=ctx.dtype),torch.as_tensor(gs,dtype=ctx.dtype),None

def layer(net,soc,kind):
    z=_Terminal.apply(net,soc,kind);return z[:,0],z[:,1],z[:,2]

def audit():
    configure();start=time.perf_counter();a=source_arrays();old=json.loads((ROOT/'results/decision_training_v1.json').read_text());pairs=np.asarray(old['train_pairs'])[:20];pv=old_signal(a,'dfl','tou');rng=np.random.default_rng(581);records=[]
    for i,(h,d) in enumerate(pairs):
        net=a['gc_shared'][h,d]-pv[h,d];actual=a['truth_gc_kw'][h,d]-a['truth_pv_kw'][h,d];soc=2.+(i%7);xx=torch.tensor(net[None],dtype=torch.float64,requires_grad=True);ss=torch.tensor([soc],dtype=torch.float64,requires_grad=True);qp=TerminalQP();buy,sell=tariff();c,dd,e=layer(xx,ss,'tou');g=torch.tensor(actual)+c-dd
        loss=.5*(torch.tensor(buy)*torch.relu(g)+torch.tensor(sell)*torch.minimum(g,torch.zeros_like(g))+.02*(c+dd)).sum()+qp.value(ss).sum()-qp.value(e[:,-1]).sum();loss.backward()
        for j in range(3):
            direction=rng.normal(size=49);direction/=np.linalg.norm(direction);analytic=float(xx.grad.numpy()[0]@direction[:48]+ss.grad.item()*direction[48]);diffs=[]
            for step in [1e-4,1e-5,1e-6]:
                vals=[]
                for sign in [-1,1]:
                    e0=soc+sign*step*direction[48];c,d2,e2,_,diag=polished_solution(qp.state(e0),net+sign*step*direction[:48]);vals.append(float(realized_bill(actual,c,d2,buy,sell)+qp.value(e0)-qp.value(e2[-1])))
                numerical=(vals[1]-vals[0])/(2*step);diffs.append({'h':step,'finite_difference':numerical,'absolute_error':abs(numerical-analytic)})
            records.append({'pair':[int(h),int(d)],'direction':j,'analytic':analytic,'scales':diffs,'minimum_error':min(v['absolute_error'] for v in diffs)})
    mx=max(r['minimum_error'] for r in records);report={'status':'PASS' if mx<1e-5 else 'FAIL','cases':20,'directions_per_case':3,'maximum_best_scale_absolute_error':mx,'records':records,**runtime_info(start)};write_json(REV/'results/continuous_gradient_audit.json',report);print(report['status'],mx,flush=True);assert report['status']=='PASS'

def twin_pairs(p,masks,tr,va):
    def keep(x,label):
        h,d=x.T;return x[(d+1<len(p['date']))&masks[label][h,np.minimum(d+1,len(p['date'])-1)]]
    return keep(tr,'train'),keep(va,'validation')

def evaluate(m,x,cap,load,actual,kind):
    m.eval();qp=TerminalQP(kind=kind);buy,sell=tariff(kind);values=[]
    with torch.no_grad():pred=m(torch.from_numpy(x.reshape(-1,774))).numpy().reshape(-1,2,48)*cap
    for j in range(len(pred)):
        e0=5.;cost=0.
        for day in range(2):c,d,e=qp.state(e0).solve(load[j,day]-pred[j,day]);cost+=float(realized_bill(actual[j,day],c,d,buy,sell));e0=e[-1]
        values.append((cost+qp.value(5.)-qp.value(e0))/2.)
    return float(np.mean(values))

def train(args):
    configure();start=time.perf_counter();p,finite,masks,tr,va=pairs_data();tr,va=twin_pairs(p,masks,tr,va)
    if args.pilot:tr=tr[:256];va=va[:128]
    a=source_arrays();
    def data(pairs):
        h,d=pairs.T;pair2=np.stack([np.stack([h,d],1),np.stack([h,d+1],1)],1);x=make_features(p,pair2.reshape(-1,2)).reshape(len(pairs),2,774);hh=pair2[:,:,0];dd=pair2[:,:,1];cap=p['capacity_kwp'][hh,dd-1,None];load=a['gc_shared'][hh,dd];actual=p['power_kw'][hh,dd,:,0]-p['power_kw'][hh,dd,:,2];target=p['power_kw'][hh,dd,:,2]/cap;return x,cap,load,actual,target
    x,cap,load,actual,target=data(tr);vx,vc,vl,vaactual,_=data(va);tx=torch.from_numpy(x);tc=torch.from_numpy(cap);tl=torch.from_numpy(load);ta=torch.from_numpy(actual);ty=torch.from_numpy(target);buy,sell=tariff(args.tariff);bt=torch.tensor(buy,dtype=torch.float32);st=torch.tensor(sell,dtype=torch.float32);qp=TerminalQP(kind=args.tariff);runs=[];epochs=1 if args.pilot else 4
    for lr in [1e-4,3e-5]:
        set_seed(11);m=initial_model('pv');opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4);gen=torch.Generator().manual_seed(11);clock=time.perf_counter();initial=evaluate(m,vx,vc,vl,vaactual,args.tariff);best=initial;state=copy.deepcopy(m.state_dict());bestepoch=0;trace=[]
        for ep in range(epochs):
            losses=[];m.train()
            for ix in torch.randperm(len(tx),generator=gen).split(32):
                out=m(tx[ix].reshape(-1,774)).reshape(-1,2,48)
                if args.loss=='mse':loss=((out-ty[ix])**2).mean()
                else:
                    soc=torch.full((len(ix),),5.,dtype=torch.float32);total=torch.zeros(len(ix))
                    for day in range(2):
                        c,d,e=layer(tl[ix,day]-out[:,day]*tc[ix,day],soc,args.tariff);g=ta[ix,day]+c-d;total+=.5*(bt*torch.relu(g)+st*torch.minimum(g,torch.zeros_like(g))+.02*(c+d)).sum(-1);soc=e[:,-1]
                    loss=((total+qp.value(torch.tensor(5.))-qp.value(soc))/2.).mean()
                opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
            value=evaluate(m,vx,vc,vl,vaactual,args.tariff);trace.append({'epoch':ep+1,'training_loss':float(np.mean(losses)),'validation_two_day_mean_boundary_adjusted_cost':value});print(args.tag,lr,trace[-1],flush=True)
            if value<best-1e-6:best=value;state=copy.deepcopy(m.state_dict());bestepoch=ep+1
        m.load_state_dict(state);runs.append((m,{'lr':lr,'initial_validation':initial,'selected_epoch':bestepoch,'best_validation_cost':best,'trace':trace,**runtime_info(clock)}))
    win=int(np.argmin([r[1]['best_validation_cost'] for r in runs]));m,log=runs[win];torch.save({'model_state_dict':m.state_dict(),'class':'PhysicalPVMLP','log':log},REV/f'models/{args.tag}.pt')
    report={'status':'complete','tag':args.tag,'loss':args.loss,'tariff':args.tariff,'seed':11,'training_anchor_pairs':tr.tolist(),'validation_anchor_pairs':va.tolist(),'n_training_anchors':len(tr),'n_validation_anchors':len(va),'unroll_days':2,'epochs':epochs,'selected':win,'candidates':[r[1] for r in runs],'terminal_value':{'lambda':qp.value_lambda,'rho':RHO,'anchor_kwh':5.},'test_used_for_selection':False,'limitation':'Truncated two-day matched training starts each training window at 5 kWh; test replays carry SOC across the complete input-valid run. It is not full-year BPTT.','resources':runtime_info(start)}
    if not args.pilot:np.savez_compressed(REV/f'forecasts/{args.tag}.npz',household_id=p['household_id'],date=p['date'],prediction_kw=forecast_all(m,p,finite,'pv'))
    write_json(REV/f'results/{args.tag}.json',report)

def replay(args):
    initialize();start=time.perf_counter();a=source_arrays();hs=np.flatnonzero(a['household_role']!=2);ds=np.flatnonzero(a['split_code']==2);pv=old_signal(a,args.model,args.tariff) if args.model in ('msecont','dfl') else np.load(REV/f'forecasts/{args.model}.npz')['prediction_kw'];gc=a['truth_gc_kw'][hs][:,ds];truthpv=a['truth_pv_kw'][hs][:,ds];cl=a['truth_cl_kw'][hs][:,ds];load=a['gc_shared'][hs][:,ds];pv=pv[hs][:,ds];shape=gc.shape;charge=np.full(shape,np.nan,np.float32);discharge=charge.copy();startstate=np.full(shape[:2],np.nan);end=startstate.copy();segment=np.full(shape[:2],-1);qp=TerminalQP(kind=args.tariff);times=[];fails=[];maxres=0.;maxcontinuity=0.;sid=0
    for i,h in enumerate(hs):
        soc=5.;previous=False
        for j,day in enumerate(ds):
            # Offline evaluation segments are defined once by observed QC,
            # identically for all arms. Future measurement magnitudes never
            # enter actions; excluded days are not secretly SOC bridges.
            if not (a['valid_input_finite'][h,day] and a['valid_history'][h,day] and a['valid_target_gcgg'][h,day]):previous=False;continue
            if not previous:soc=5.;sid+=1
            startstate[i,j]=soc;segment[i,j]=sid;clock=time.perf_counter()
            try:
                c,d,e,diag=qp.state(soc).solve(load[i,j]-pv[i,j],True);times.append(time.perf_counter()-clock);charge[i,j]=c;discharge[i,j]=d;end[i,j]=e[-1];maxres=max(maxres,diag['energy_balance_max']);
                if previous:maxcontinuity=max(maxcontinuity,abs(startstate[i,j]-end[i,j-1]))
                soc=e[-1];previous=True
            except Exception as exc:fails.append({'household_id':int(a['household_id'][h]),'date':str(a['date'][day]),'error':str(exc)});previous=False
        if (i+1)%50==0:print(args.tag,i+1,'/',len(hs),time.perf_counter()-start,flush=True)
    buy,sell=tariff(args.tariff);throughput=.5*(charge+discharge).sum(-1);cost=realized_bill(gc-truthpv,charge,discharge,buy,sell);bill=cost+.09*cl.sum(-1);boundary=qp.value(startstate)-qp.value(end);valid=a['valid_history'][hs][:,ds]&a['valid_target_gcgg'][hs][:,ds];clean=a['valid_history'][hs][:,ds]&a['valid_target_feeder'][hs][:,ds]
    np.savez_compressed(REV/f'dispatch/{args.tag}.npz',household_id=a['household_id'][hs],household_role=a['household_role'][hs],date=a['date'][ds],charge_kw=charge,discharge_kw=discharge,bill_aud=bill,operating_excluding_cl_aud=cost,retail_bill_aud=bill-.02*throughput,energy_throughput_kwh=throughput,clean_day=clean,valid_gcgg_day=valid,actual_load_kw=gc+cl,actual_pv_kw=truthpv,forecast_load_kw=load,forecast_pv_kw=pv,forecast_net_kw=load-pv,start_energy_kwh=startstate,end_energy_kwh=end,segment_id=segment,boundary_inventory_adjustment_aud=boundary,boundary_adjusted_operating_aud=cost+boundary)
    report={'status':'PASS' if not fails else 'FAIL','tag':args.tag,'failures':fails,'soc_carry_error_kwh':maxcontinuity,'max_energy_equation_residual_kwh':maxres,'segment_count':sid,'finite_household_days':int(np.isfinite(end).sum()),'terminal_value':{'lambda':qp.value_lambda,'rho':RHO,'anchor_kwh':5.},'mean_raw_operating_excluding_cl':float(cost[valid].mean()),'mean_boundary_adjusted_operating':float((cost+boundary)[valid].mean()),'interpretation':'Report raw operating cost and inventory-adjusted cost separately; daily value changes telescope only across contiguous complete segments. Clean-mask exclusion can break exact telescoping for the clean-only summary.','mean_end_energy_kwh':float(end[valid].mean()),'solve_latency_seconds':{'mean':float(np.mean(times)),'median':float(np.median(times)),'p95':float(np.quantile(times,.95))},**runtime_info(start)};write_json(REV/f'dispatch/{args.tag}.json',report);assert not fails and np.isfinite(cost[valid]).all();print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['audit','train','replay'],required=True);ap.add_argument('--loss',choices=['mse','dfl'],default='mse');ap.add_argument('--tariff',choices=['tou','flat'],default='tou');ap.add_argument('--model',default='msecont');ap.add_argument('--tag',default='continuous_pilot');ap.add_argument('--pilot',action='store_true');args=ap.parse_args();{'audit':audit,'train':lambda:train(args),'replay':lambda:replay(args)}[args.stage]()
