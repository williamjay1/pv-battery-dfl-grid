"""Paired all-household economic comparison, preserving the calendar axis."""
from revision_forecast_common import *
import argparse

def uncertainty(delta,mask,rng,nboot=2000):
    counts=mask.sum(0);sums=np.where(mask,delta,0).sum(0);nd=delta.shape[1];boot=[]
    for _ in range(nboot):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/7)));ix=((starts[:,None]+np.arange(7))%nd).ravel()[:nd];boot.append(sums[ix].sum()/counts[ix].sum())
    return np.quantile(boot,[.025,.975]).tolist()

def macro_uncertainty(delta,mask,seed=9417,nboot=2000):
    rng=np.random.default_rng(seed);nd=delta.shape[1];starts=rng.integers(0,nd,size=(nboot,int(np.ceil(nd/7))));ix=((starts[:,:,None]+np.arange(7))%nd).reshape(nboot,-1)[:,:nd];weights=np.zeros((nboot,nd),float);np.add.at(weights,(np.arange(nboot)[:,None],ix),1)
    keep=mask.sum(1)>0;den=mask[keep].astype(float)@weights.T;num=np.where(mask[keep],delta[keep],0)@weights.T;values=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);return np.quantile(np.nanmean(values,axis=0),[.025,.975]).tolist()

def main(args):
    initialize();st=time.perf_counter();names=args.methods.split(',');arrays={n:dict(np.load(REV/f'dispatch/{n}.npz')) for n in names};ref=arrays[names[0]]
    wear={n:json.loads((REV/f'dispatch/{n}.json').read_text()).get('battery',{}).get('throughput',.02) for n in names}
    for z in arrays.values():assert np.array_equal(z['household_id'],ref['household_id']) and np.array_equal(z['date'],ref['date'])
    masks=np.logical_and.reduce([z['valid_gcgg_day']&np.isfinite(z['operating_excluding_cl_aud']) for z in arrays.values()]);full=np.logical_and.reduce([z['clean_day']&np.isfinite(z['bill_aud']) for z in arrays.values()]);out={'methods':names,'reference':names[0],'mask_policy':'same individual history+GC/PV household days for paired economics; full retail requires valid CL too','currency':'AUD per household-day','resampling':'paired circular 7-calendar-day blocks, 2000 replicates, fixed seed9417; never treats replicas as new tests','cohorts':{}}
    for cohort,hh in [('all299',np.ones(len(ref['household_id']),bool)),('known244',ref['household_role']==0),('heldout55',ref['household_role']==1)]:
        mask=masks&hh[:,None];fm=full&hh[:,None];r={'n_households':int(hh.sum()),'gcgg_household_days':int(mask.sum()),'fullcost_household_days':int(fm.sum()),'absolute':{},'paired':{}}
        for name,z in arrays.items():
            r['absolute'][name]={'mean_operating_excluding_cl':float(z['operating_excluding_cl_aud'][mask].mean()),'mean_full_operating_cost':float(z['bill_aud'][fm].mean()),'mean_retail_only':float(z['retail_bill_aud'][fm].mean()),'mean_throughput_kwh':float(z['energy_throughput_kwh'][mask].mean())}
        for i,name in enumerate(names):
            for base in names[:i]:
                delta=arrays[name]['operating_excluding_cl_aud']-arrays[base]['operating_excluding_cl_aud'];house=np.divide(np.where(mask,delta,0).sum(1),mask.sum(1),out=np.full(len(hh),np.nan),where=mask.sum(1)>0);house=house[hh];micro=float(delta[mask].mean())
                wd=wear[name]*arrays[name]['energy_throughput_kwh']-wear[base]*arrays[base]['energy_throughput_kwh']
                r['paired'][name+'__minus__'+base]={'micro_mean_difference':micro,'micro_calendar_block95':uncertainty(delta,mask,np.random.default_rng(9417)),'macro_mean_difference':float(np.nanmean(house)),'macro_calendar_block95':macro_uncertainty(delta,mask),'household_median_difference':float(np.nanmedian(house)),'household_p10_p90':np.nanquantile(house,[.1,.9]).tolist(),'fraction_households_benefit':float(np.nanmean(house<0)),'retail_component_difference':float((delta-wd)[mask].mean()),'wear_component_difference':float(wd[mask].mean())}
        out['cohorts'][cohort]=r
    out['resource_files']={n:str(REV/f'dispatch/{n}.json') for n in names};out['resources']=runtime_info(st);write_json(REV/f'results/{args.tag}.json',out);print(json.dumps(out['cohorts']['all299']['paired']),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--methods',required=True);ap.add_argument('--tag',required=True);main(ap.parse_args())
