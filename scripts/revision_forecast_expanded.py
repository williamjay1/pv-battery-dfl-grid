"""Seed-11 matched all-available-household economic expansion; no fitting."""
import argparse,json,time
from revision_forecast_common import *

def main(args):
    initialize();a=source_arrays();scenarios=None;pv=old_signal(a,args.method,args.tariff)
    if args.method=='stochastic20':
        from decision_baselines import residual_bank
        hs=np.flatnonzero(a['household_role']!=2);t=time.perf_counter();rg,rp,bank=residual_bank(a,hs)
        prior=json.loads((ROOT/f'results/decision_baselines_physical_stochastic_bank_{args.tariff}.json').read_text())
        donor_map=dict(zip(prior['target_household_ids'],prior['donor_household_ids']));idpos={int(v):i for i,v in enumerate(a['household_id'])};days=np.array([np.flatnonzero(a['date']==date)[0] for date in prior['scenario_dates']])
        assert bank['scenario_dates']==prior['scenario_dates']
        for i,h in enumerate(hs):
            if a['household_role'][h]!=1:continue
            donor=idpos[donor_map[int(a['household_id'][h])]];bank['donor_household_ids'][i]=int(a['household_id'][donor])
            rg[i]=a['truth_gc_kw'][donor,days]-a['gc_shared'][donor,days]
            rp[i]=(a['truth_pv_kw'][donor,days]-a['pv_mlp_seed11'][donor,days])/a['capacity_kwp'][donor,days,None]
        rp[:,:,NIGHT]=0;scenarios=(rg,rp)
        bank['preparation_resources']=runtime_info(t);write_json(REV/'results'/f'expanded_stochastic_bank_{args.tariff}.json',bank)
    suffix=f'_pilot{args.pilot}' if args.pilot else ''
    original={'calibrated3':'calibrated','stochastic20':'stochastic'}.get(args.method,args.method)
    reuse=None if args.pilot else ROOT/f'results/dispatch/{original}_{args.tariff}_seed11_test_physical.npz'
    dispatch(a,f'{args.method}_{args.tariff}_seed11_expanded{suffix}',args.tariff,pv_signal=pv,scenarios=scenarios,limit_pairs=args.pilot,reuse_path=reuse)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['msecont','dfl','calibrated3','stochastic20'],required=True);p.add_argument('--tariff',choices=['tou','flat'],default='tou');p.add_argument('--pilot',type=int);main(p.parse_args())
