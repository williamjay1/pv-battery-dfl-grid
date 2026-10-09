"""Independent real-data finite-difference check of the differentiable QP."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from battery_model import DispatchQP,tariff,realized_bill
from decision_layer import BatteryLayer,polished_solution
ROOT=Path(__file__).resolve().parents[1]

def audit(count=24,output='forecast_battery_gradient_review_polished.json'):
    torch.set_num_threads(2)
    z=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    mask=z['valid_history']&z['valid_target_gcgg']&(z['household_role']==0)[:,None]&(z['split_code']==0)[None,:]
    allpairs=np.argwhere(mask);pairs=allpairs[np.linspace(0,len(allpairs)-1,count,dtype=int)]
    net=z['gc_shared'][pairs[:,0],pairs[:,1]].astype(float)-z['pv_mlp_seed11'][pairs[:,0],pairs[:,1]].astype(float)
    actual=z['truth_gc_kw'][pairs[:,0],pairs[:,1]].astype(float)-z['truth_pv_kw'][pairs[:,0],pairs[:,1]].astype(float)
    rng=np.random.default_rng(90173);records=[];failures=[];start=time.perf_counter()
    for kind in ['tou','flat','equal']:
        qp=DispatchQP(kind=kind,tolerance=1e-13);buy,sell=tariff(kind);layer=BatteryLayer(kind=kind)
        def forward(row):
            if kind=='equal':return qp.solve(row)
            c,d,e,_,_=polished_solution(qp,row);return c,d,e
        x=torch.tensor(net,dtype=torch.float64,requires_grad=True)
        try:
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');c,d,e=layer(x);g=torch.tensor(actual)+c-d
                loss=.5*(torch.tensor(buy)*torch.relu(g)+torch.tensor(sell)*torch.minimum(g,torch.zeros_like(g))+.02*(c+d)).sum();loss.backward()
            grads=x.grad.detach().numpy();batch_warnings=[str(w.message) for w in ww]
        except Exception as err:
            failures.append({'tariff':kind,'stage':'batch_gradient','error':repr(err)});continue
        for i,row in enumerate(net):
            try:
                cc,dd,ee=forward(row)
                single=torch.tensor(row,dtype=torch.float64,requires_grad=True);sc,sd,se=layer(single);sg=torch.tensor(actual[i])+sc-sd
                sl=.5*(torch.tensor(buy)*torch.relu(sg)+torch.tensor(sell)*torch.minimum(sg,torch.zeros_like(sg))+.02*(sc+sd)).sum();sl.backward()
                consistency=float(np.max(abs(single.grad.detach().numpy()-grads[i])))
                rec={'tariff':kind,'household_id':int(z['household_id'][pairs[i,0]]),'date':str(z['date'][pairs[i,1]]),'batch_single_gradient_max_gap':consistency,'gradient_l2':float(np.linalg.norm(grads[i])),'actual_grid_min_abs_kw':float(np.min(abs(actual[i]+cc-dd))),'batch_warnings':batch_warnings,'directions':[]}
                for j in range(3):
                    direction=rng.normal(size=48);direction/=np.linalg.norm(direction);ad=float(grads[i]@direction);vals=[]
                    for h in [1e-3,1e-4,1e-5]:
                        cp,dp,_=forward(row+h*direction);cm,dm,_=forward(row-h*direction)
                        fd=float((realized_bill(actual[i],cp,dp,buy,sell)-realized_bill(actual[i],cm,dm,buy,sell))/(2*h));err=abs(fd-ad)
                        vals.append({'step':h,'central_difference':fd,'absolute_error':err,'scaled_relative_error':err/max(1e-3,abs(fd),abs(ad))})
                    rec['directions'].append({'adjoint':ad,'step_checks':vals})
                records.append(rec)
            except Exception as err:
                failures.append({'tariff':kind,'stage':'sample_gradient','sample':i,'error':repr(err)})
    checks=[v for a in records for b in a['directions'] for v in b['step_checks'] if v['step']==1e-4]
    small=[v for a in records for b in a['directions'] for v in b['step_checks'] if v['step']==1e-5]
    mx=max((v['absolute_error'] for v in small),default=float('inf'))
    out={'status':'PASS' if not failures and mx<1e-5 else 'REVIEW','scope':f'{count} real clean training household-days per tariff, three deterministic random directions, three finite-difference steps, no skipped samples','actual_settlement':'metered GC minus metered PV; predicted input shared GC minus seed11 PV','runtime_seconds':time.perf_counter()-start,'failures':failures,'max_absolute_error_step1e4':max((v['absolute_error'] for v in checks),default=None),'max_scaled_relative_error_step1e4':max((v['scaled_relative_error'] for v in checks),default=None),'max_absolute_error_step1e5':mx,'max_scaled_relative_error_step1e5':max((v['scaled_relative_error'] for v in small),default=None),'max_batch_single_gradient_gap':max((a['batch_single_gradient_max_gap'] for a in records),default=None),'records':records}
    (ROOT/'results'/output).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=24);ap.add_argument('--output',default='forecast_battery_gradient_review_polished.json');args=ap.parse_args();audit(args.count,args.output)
