"""Conditional-control contrasts, including honest inventory accounting."""
from revision_forecast_common import *
from revision_forecast_compare import uncertainty,macro_uncertainty

def contrast(left,right,key='operating_excluding_cl_aud'):
    assert np.array_equal(left['household_id'],right['household_id']) and np.array_equal(left['date'],right['date']);mask=left['valid_gcgg_day']&right['valid_gcgg_day']&np.isfinite(left[key])&np.isfinite(right[key]);delta=left[key]-right[key];out={}
    for label,hh in [('all299',np.ones(len(mask),bool)),('known244',left['household_role']==0),('heldout55',left['household_role']==1)]:
        mm=mask&hh[:,None];hm=np.divide(np.where(mm,delta,0).sum(1),mm.sum(1),out=np.full(len(mask),np.nan),where=mm.sum(1)>0);out[label]={'household_days':int(mm.sum()),'micro_mean_difference_aud':float(delta[mm].mean()),'macro_mean_difference_aud':float(np.nanmean(hm)),'calendar_block95':uncertainty(delta,mm,np.random.default_rng(9417)),'macro_calendar_block95':macro_uncertainty(delta,mm),'fraction_households_benefit':float((hm[hh]<0).mean())}
    return out

def main():
    initialize();out={'status':'partial','currency':'AUD per household-day; negative DFL-minus-MSE favors DFL','selection':'post-freeze descriptive contrasts; no test-selected models or settings','load':{},'wear':{},'continuous':{}};load=lambda n:dict(np.load(REV/f'dispatch/{n}.npz'))
    ready=lambda n:(REV/f'dispatch/{n}.npz').exists() and (REV/f'dispatch/{n}.json').exists()
    for setting in ['persistence','strong']:
        r={}
        for stage in ['frozen','matched']:
            names=[f'frozen_{m}_load{setting}_tou' if stage=='frozen' else f'matched_pv_{setting}_{"mse" if m=="msecont" else "dfl"}_tou_wear002_expanded' for m in ['msecont','dfl']]
            if all(ready(n) for n in names):r[stage]=contrast(load(names[1]),load(names[0]));r[stage+'_files']=names
        out['load'][setting]=r
    base=[load(f'{m}_tou_seed11_expanded') for m in ['msecont','dfl']]
    for coeff in [0.,.01,.04]:
        label='0' if coeff==0 else str(coeff);r={};repriced=[]
        for z in base:
            tmp=z.copy();tmp['repriced']=z['operating_excluding_cl_aud']+(coeff-.02)*z['energy_throughput_kwh'];repriced.append(tmp)
        r['frozen_actions_repriced']=contrast(repriced[1],repriced[0],'repriced')
        for stage in ['frozen','matched']:
            names=[f'frozen_{m}_tou_wear{label}' if stage=='frozen' else f'matched_pv_hgb_{"mse" if m=="msecont" else "dfl"}_tou_wear{label}_expanded' for m in ['msecont','dfl']]
            if all(ready(n) for n in names):r[stage+'_signals_reoptimized']=contrast(load(names[1]),load(names[0]));r[stage+'_files']=names
        out['wear'][label]=r
    for stage in ['frozen','matched']:
        names=[f'continuous_frozen_{m}_tou' if stage=='frozen' else f'continuous_matched_{"mse" if m=="msecont" else "dfl"}_tou_expanded' for m in ['msecont','dfl']]
        if not all(ready(n) for n in names):continue
        zz=[load(n) for n in names];assert np.array_equal(zz[0]['segment_id'],zz[1]['segment_id']);r={'files':names,'raw_operating_contrast':contrast(zz[1],zz[0]),'inventory_adjusted_contrast':contrast(zz[1],zz[0],'boundary_adjusted_operating_aud'),'arms':{}}
        for name,z in zip(names,zz):
            mask=z['valid_gcgg_day'];lam=(.18+.02)/.95;linear=lam*(z['start_energy_kwh']-z['end_energy_kwh']);quad=-.01*((z['start_energy_kwh']-5)**2-(z['end_energy_kwh']-5)**2);assert np.nanmax(abs(linear+quad-z['boundary_inventory_adjustment_aud']))<1e-10
            r['arms'][name]={'household_days':int(mask.sum()),'segment_count':int(len(np.unique(z['segment_id'][mask]))),'mean_raw_operating_cost':float(z['operating_excluding_cl_aud'][mask].mean()),'mean_linear_inventory_adjustment':float(linear[mask].mean()),'mean_quadratic_inventory_adjustment':float(quad[mask].mean()),'mean_inventory_adjusted_cost':float(z['boundary_adjusted_operating_aud'][mask].mean()),'mean_daily_end_energy':float(z['end_energy_kwh'][mask].mean())}
        out['continuous'][stage]=r
    complete=all(len(out['load'][s])==4 for s in out['load']) and all('matched_signals_reoptimized' in r and 'frozen_signals_reoptimized' in r for r in out['wear'].values()) and len(out['continuous'])==2;out['status']='complete' if complete else 'partial';write_json(REV/'results/control_transfer_summary.json',out);print(json.dumps({'status':out['status'],'load':{k:list(v) for k,v in out['load'].items()},'wear':{k:list(v) for k,v in out['wear'].items()},'continuous':list(out['continuous'])}),flush=True)

if __name__=='__main__':main()
