"""Identical validation-day single-thread dispatch timing benchmark."""
from revision_forecast_common import *
import argparse
from decision_baselines import residual_bank

def main(args):
    initialize();start=time.perf_counter();a=source_arrays();pairs=np.asarray(json.loads((ROOT/'results/decision_training_v1.json').read_text())['validation_pairs']);h,d=pairs.T;load=a['gc_shared'][h,d];pv=old_signal(a,args.method,args.tariff)[h,d];net=load-pv;prep=time.perf_counter();scenario_info=None
    if args.method=='stochastic20':
        hs=np.flatnonzero(a['household_role']==0);gc,rp,scenario_info=residual_bank(a,hs);index={v:i for i,v in enumerate(hs)};ix=np.asarray([index[v] for v in h]);paths=np.maximum(pv[:,None,:]+rp[ix]*a['capacity_kwp'][h,d-1,None,None],0);paths[:,:,NIGHT]=0;net=np.maximum(load[:,None,:]+gc[ix],0)-paths
    preparation=time.perf_counter()-prep;qp=DispatchQP(kind=args.tariff,n_scenarios=20 if args.method=='stochastic20' else 1,tolerance=1e-13)
    for row in net[:20]:qp.solve(row)
    wall=[];cpu=[];checks=[]
    for repeat in range(3):
        checksum=0.
        for row in net:
            tc=time.process_time();tw=time.perf_counter();c,dd,e=qp.solve(row);wall.append(time.perf_counter()-tw);cpu.append(time.process_time()-tc);checksum+=c.sum()+2*dd.sum()
        checks.append(checksum)
    assert max(checks)-min(checks)<1e-5
    report={'method':args.method,'tariff':args.tariff,'seed':11,'sample_source':'original 1024 known-household validation pairs in decision_training_v1.json, same exact order for all methods','n_pairs':len(pairs),'repeats':3,'warmup_calls':20,'timing_scope':'QP solve including RHS update; excludes network feature/inference and data import','preparation_seconds':preparation,'scenario_info':scenario_info,'mean_qp_ms':float(np.mean(wall)*1000),'median_qp_ms':float(np.median(wall)*1000),'p95_qp_ms':float(np.quantile(wall,.95)*1000),'mean_process_cpu_qp_ms':float(np.mean(cpu)*1000),'deterministic_checksums':checks,**runtime_info(start)};write_json(REV/f'results/benchmark_{args.method}_{args.tariff}.json',report);print(json.dumps({k:v for k,v in report.items() if k!='scenario_info'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['msecont','dfl','calibrated3','stochastic20'],required=True);ap.add_argument('--tariff',choices=['tou','flat'],default='tou');main(ap.parse_args())
