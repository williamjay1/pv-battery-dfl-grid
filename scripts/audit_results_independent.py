"""Read-only independent arithmetic audit; never fits/selects any controller."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_network

def prices(kind):
    hour=np.arange(48)/2
    buy=np.full(48,.30)
    if kind=='tou':
        buy[(hour<7)|(hour>=22)]=.18
        buy[(hour>=16)&(hour<21)]=.50
    return buy,np.full(48,.08)

def cost(gc,pv,cl,c,d,kind):
    buy,sell=prices(kind)
    g=np.asarray(gc,dtype=float)-pv+np.asarray(c,dtype=float)-d
    retail=.5*(np.where(g>=0,buy*g,sell*g).sum(-1)+.18*np.asarray(cl,dtype=float).sum(-1))
    wear=.01*(np.asarray(c,dtype=float)+d).sum(-1)
    return retail,wear

def independent_lp(gc,pv,cl,kind):
    n=48;buy,sell=prices(kind);net=np.asarray(gc,dtype=float)-pv
    objective=np.r_[.5*(sell+.02),.5*(-sell+.02),np.zeros(n),.5*(buy-sell)]
    eq=np.zeros((n+1,4*n));rhs=np.zeros(n+1);rhs[0]=5;rhs[-1]=5
    for t in range(n):
        eq[t,t]=-.5*.95;eq[t,n+t]=.5/.95;eq[t,2*n+t]=1
        if t:eq[t,2*n+t-1]=-1
    eq[-1,3*n-1]=1
    ine=np.zeros((n,4*n))
    for t in range(n):ine[t,t]=1;ine[t,n+t]=-1;ine[t,3*n+t]=-1
    solved=linprog(objective,A_ub=ine,b_ub=-net,A_eq=eq,b_eq=rhs,
                   bounds=[(0,3)]*(2*n)+[(1,9)]*n+[(0,None)]*n,method='highs')
    if not solved.success:raise RuntimeError(solved.message)
    return float(solved.fun+.5*np.dot(sell,net)+.09*np.sum(cl))

def main():
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text())
    source=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    ids=source['household_id'];dates=source['date'];test=np.flatnonzero(source['split_code']==2)
    gc=source['truth_gc_kw'][:,test];pv=source['truth_pv_kw'][:,test];cl=source['truth_cl_kw'][:,test]
    groups=splits['primary_panels']+[splits['heldout_network_household_ids']]
    actual_reports=json.loads((ROOT/'results/analysis/dispatch_cost_components.json').read_text())['records']
    saved={(r['model'],r['tariff'],r['seed'],r['panel']):r for r in actual_reports}
    outputs=[];oracle_cache={};sample_pairs=[]
    for panel,group in enumerate(groups):
        clean_dates=splits['primary_panel_common_clean_dates'][panel]['test'] if panel<3 else splits['heldout_network_common_clean_dates']['test']
        selected=np.flatnonzero(np.isin(dates[test],clean_dates))
        for hid in [group[0],group[len(group)//2],group[-1]]:
            for day in selected[[0,len(selected)//2,-1]]:sample_pairs.append((int(hid),int(day)))
    report={'audit_type':'fixed deterministic arithmetic/feasibility checks, not model selection',
            'source_hashes':{n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in ['analyze_dispatch.py','analyze_network.py']}}
    for file in sorted((ROOT/'results/dispatch').glob('*_test_tight.npz')):
        meta_file=file.with_suffix('.json')
        if not meta_file.exists():continue
        meta=json.loads(meta_file.read_text());kind=meta['tariff'];model=meta.get('model',meta.get('method'));seed=meta['seed']
        z=np.load(file);zi=z['household_id'];h=np.array([np.flatnonzero(ids==i)[0] for i in zi]);look={int(i):j for j,i in enumerate(zi)}
        if not np.array_equal(z['date'],dates[test]):raise AssertionError('Date axis mismatch')
        if kind not in oracle_cache:oracle_cache[kind]=np.load(ROOT/f'results/dispatch/oracle_{kind}_seed11_test_tight.npz')
        oracle=oracle_cache[kind]
        assert np.array_equal(zi,oracle['household_id']) and np.array_equal(z['date'],oracle['date'])
        c=z['charge_kw'];d=z['discharge_kw'];bill=z['bill_aud'];retail,wear=cost(gc[h],pv[h],cl[h],c,d,kind)
        mask=z['clean_day'];expected=source['valid_history'][h][:,test]&source['valid_target_feeder'][h][:,test]
        assert np.array_equal(mask,expected)
        assert np.all(np.isfinite(bill[mask])) and np.isfinite(c[mask]).all() and np.isfinite(d[mask]).all()
        err=float(np.max(abs((retail+wear-bill)[mask])))
        assert err<3e-6,(file.name,err)
        records=[]
        for panel,group in enumerate(groups):
            rows=np.array([look[int(i)] for i in group]);valid=mask[rows].all(0)
            assert len(rows)==55
            expected_dates=splits['primary_panel_common_clean_dates'][panel]['test'] if panel<3 else splits['heldout_network_common_clean_dates']['test']
            assert np.array_equal(valid,np.isin(z['date'],expected_dates))
            assert np.array_equal(valid,oracle['clean_day'][rows].all(0))
            denominator=len(rows)*int(valid.sum())
            primary=float(bill[rows][:,valid].sum()/denominator)
            r={'panel':panel,'household_days':denominator,'n_clean_days':int(valid.sum()),'operating_cost':primary,
               'retail_independent':float(retail[rows][:,valid].sum()/denominator),
               'wear_independent':float(wear[rows][:,valid].sum()/denominator),
               'minimum_oracle_difference':float((bill[rows][:,valid]-oracle['bill_aud'][rows][:,valid]).min())}
            key=(model,kind,seed,panel)
            if key in saved:
                s=saved[key];r['reported_mean_difference']=abs(primary-s['operating_cost_aud_per_household_day'])
                assert r['reported_mean_difference']<1e-12
                assert abs(r['retail_independent']-s['retail_electricity_bill_aud_per_household_day'])<3e-6
                assert abs(r['wear_independent']-s['throughput_proxy_aud_per_household_day'])<3e-6
            if model not in ['self','none']:assert r['minimum_oracle_difference']>=-1e-5
            records.append(r)
        outputs.append({'file':file.name,'model':model,'tariff':kind,'seed':seed,'clean_household_days':int(mask.sum()),'max_independent_cost_error_aud':err,'panels':records})
    report['dispatch']=outputs
    lp=[]
    for kind in ['tou','flat']:
        oracle=oracle_cache[kind];look={int(i):j for j,i in enumerate(oracle['household_id'])}
        for hid,day in [sample_pairs[9*panel+4*j] for panel in range(4) for j in range(3)]:
            h=int(np.flatnonzero(ids==hid)[0]);minimum=independent_lp(gc[h,day],pv[h,day],cl[h,day],kind)
            actual=float(oracle['bill_aud'][look[hid],day]);lp.append({'tariff':kind,'household':hid,'date':str(dates[test[day]]),'unregularized_lp':minimum,'regularized_oracle':actual,'oracle_minus_lp':actual-minimum})
    report['independent_lp']=lp
    networks=[]
    for panel in range(3):
        for method in ['mlp','dfl']:
            stem=f'{method}_seed11_panel{panel}_b28_all_tou_eulv_src1p05_fixed_test_tight'
            file=ROOT/'results/network_replay/main_v1'/f'{stem}.npz';st=file.with_suffix('.json')
            if not st.exists() or json.loads(st.read_text()).get('status')!='completed':continue
            z=np.load(file);metric=analyze_network.metrics(z);valid=z['clean_day']&z['converged_day'];n=len(z['date'])
            assert n==365 and np.all(np.diff(z['date'].astype('datetime64[D]')).astype(int)==1)
            assert np.isfinite(metric['bill_aud']).sum()==int(valid.sum())
            src=np.load(ROOT/f'results/dispatch/{method}_tou_seed11_test_tight.npz');none=np.load(ROOT/'results/dispatch/none_tou_seed11_test_tight.npz')
            pos={int(v):i for i,v in enumerate(src['household_id'])};h=[pos[int(v)] for v in z['household_id']]
            expected_bill=np.concatenate([src['bill_aud'][h[:28]],none['bill_aud'][h[28:]]]).mean(0)
            err=float(np.max(abs(metric['bill_aud'][valid]-expected_bill[valid])));assert err<1e-12
            vh=z['household_voltage_pu'].astype(float).reshape(n,48,55)
            independent_excursion=np.maximum(.95-vh,0).mean((1,2))+np.maximum(vh-1.05,0).mean((1,2))
            verr=float(np.max(abs(independent_excursion[valid]-metric['mean_voltage_excursion_pu'][valid])))
            assert verr<1e-7
            independent_fraction=100*((vh<.95)|(vh>1.05)).mean((1,2))
            near=((abs(vh-.95)<1e-7)|(abs(vh-1.05)<1e-7)).sum((1,2))*100/(48*55)
            fracerr=abs(independent_fraction-metric['voltage_percentage_points'])
            assert np.all(fracerr[valid]<=near[valid]+1e-10)
            networks.append({'file':file.name,'clean_days':int(valid.sum()),'cost_error':err,'voltage_excursion_error_due_to_float32_storage':verr,'voltage_fraction_max_pp_difference':float(np.nanmax(fracerr)),'denominator':55*48})
    report['network_metric_checks']=networks
    paired=[]
    for panel in range(3):
        folder=ROOT/'results/network_replay/main_v1'
        files=[folder/f'{m}_seed11_panel{panel}_b28_all_tou_eulv_src1p05_fixed_test_tight.npz' for m in ['mlp','dfl']]
        if not all(f.exists() and json.loads(f.with_suffix('.json').read_text()).get('status')=='completed' for f in files):continue
        metrics=[analyze_network.metrics(np.load(f)) for f in files]
        x=metrics[1]['bill_aud']-metrics[0]['bill_aud'];idx=analyze_network.bootstrap_indices(365,7)
        boot=np.array([np.sum(x[row][np.isfinite(x[row])])/np.isfinite(x[row]).sum() for row in idx])
        assert np.isfinite(boot).all()
        s=analyze_network.summarize(x,idx)
        assert np.allclose(s['ci95'],np.percentile(boot,[2.5,97.5]),rtol=0,atol=1e-12)
        paired.append({'panel':panel,'orientation':'DFL minus MLP','seed':11,'days':s['n_clean_days'],'mean_delta':s['mean'],'ci95_independent':s['ci95'],'bootstrap_draws':len(idx),'all_bootstraps_have_valid_days':True,'calendar_axis_preserved':len(x)==365})
    report['paired_bootstrap_checks']=paired
    report['status']='PASS_arithmetic_with_documented_scope_limits'
    (ROOT/'results/forecast_results_independent_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'dispatch_outputs':len(outputs),'panel_records':sum(len(x['panels']) for x in outputs),'max_cost_error':max(x['max_independent_cost_error_aud'] for x in outputs),'lp_checks':len(lp),'min_oracle_minus_lp':min(r['oracle_minus_lp'] for r in lp),'max_oracle_minus_lp':max(r['oracle_minus_lp'] for r in lp),'network_checks':len(networks),'bootstrap_checks':len(paired)},indent=2))

if __name__=='__main__':main()
