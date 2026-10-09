"""Shared revision-only data, accounting, resources and dispatch contract.

All target measurements enter settlement only. Forecast inputs are frozen before
the target day. Original project artifacts are read-only to this module.
"""
from pathlib import Path
import os,time,json,platform,ctypes
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'revision_20261009'
os.environ['TEMP']=str(ROOT/'temp');os.environ['TMP']=str(ROOT/'temp')
import numpy as np
from battery_model import Battery,DispatchQP,tariff,realized_bill

NIGHT=np.r_[0:8,42:48]

def initialize():
    for name in ['dispatch','forecasts','models','results','logs','controls']:(REV/name).mkdir(parents=True,exist_ok=True)

def peak_rss_bytes():
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        c=Counters();c.cb=ctypes.sizeof(c)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        handle=ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):raise OSError('Cannot read process RSS')
        return int(c.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def runtime_info(start):
    return {'elapsed_seconds':time.perf_counter()-start,'peak_process_rss_bytes':peak_rss_bytes(),'logical_threads_per_process':1,'python':platform.python_version(),'platform':platform.platform(),'resource_note':'Peak process RSS includes data and imports. Wall times are comparable only when run serially without competing workload.'}

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def source_arrays():
    z=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    names=['household_id','household_role','date','split_code','valid_history','valid_target_gcgg','valid_target_feeder','valid_input_finite','gc_shared','pv_mlp_seed11','capacity_kwp','truth_gc_kw','truth_pv_kw','truth_cl_kw']
    a={k:z[k] for k in names};finite=np.isfinite(a['pv_mlp_seed11']).all(-1)
    for slot in NIGHT:a['pv_mlp_seed11'][:,:,slot][finite]=0
    return a

def old_signal(a,method,kind):
    if method=='msecont':
        return np.load(ROOT/'datasets/mse_continuation_arrays_physical.npz')[f'pv_msecont_{kind}_seed11']
    if method=='dfl':return np.load(ROOT/'datasets/decision_arrays_physical.npz')[f'pv_dfl_{kind}_seed11']
    if method=='calibrated3':
        from decision_baselines import BANDS
        pars=json.loads((ROOT/f'results/decision_baselines_physical_calibrated_{kind}.json').read_text())
        return np.maximum(a['pv_mlp_seed11']+a['capacity_kwp'][:,:,None]*(np.asarray(pars['selected_bias_kw_per_kwp'])@BANDS),0)
    if method in ['stochastic20','frozen']:return a['pv_mlp_seed11'].copy()
    raise KeyError(method)

def dispatch(a,name,kind,pv_signal=None,net_signal=None,load_signal=None,battery=Battery(),scenarios=None,period=2,limit_pairs=None,reuse_path=None):
    """Expand to every nonexcluded ID; feeder and single-household masks coexist."""
    initialize();start=time.perf_counter();hs=np.flatnonzero(a['household_role']!=2);ds=np.flatnonzero(a['split_code']==period)
    take=lambda v:np.asarray(v)[hs][:,ds]
    gc=take(a['truth_gc_kw']);pv=take(a['truth_pv_kw']);cl=take(a['truth_cl_kw'])
    load=take(a['gc_shared'] if load_signal is None else load_signal)
    forecast=take(a['pv_mlp_seed11'] if pv_signal is None else pv_signal)
    net=load-forecast if net_signal is None else take(net_signal)
    valid=take(a['valid_history']&a['valid_target_feeder']);single=take(a['valid_history']&a['valid_target_gcgg'])
    inp=take(a['valid_input_finite'])&np.isfinite(net).all(-1)
    c=np.full_like(gc,np.nan);d=c.copy();end=np.full(gc.shape[:2],np.nan);bill=end.copy();excl=end.copy();retail=end.copy();throughput=end.copy()
    qp=DispatchQP(battery,kind,n_scenarios=20 if scenarios is not None else 1,tolerance=1e-13);buy,sell=tariff(kind)
    failures=[];solve_times=[];residual=0.;maxsim=0.;count=0;reused=np.zeros(end.shape,bool)
    if reuse_path is not None:
        old=np.load(reuse_path);assert np.array_equal(old['date'],a['date'][ds])
        lookup={int(v):i for i,v in enumerate(a['household_id'][hs])}
        for k,hid in enumerate(old['household_id']):
            i=lookup[int(hid)];ok=np.isfinite(old['charge_kw'][k]).all(-1)
            assert np.array_equal(old['forecast_pv_kw'][k][ok],forecast[i][ok])
            assert np.array_equal(old['forecast_load_kw'][k][ok],load[i][ok])
            c[i,ok]=old['charge_kw'][k,ok];d[i,ok]=old['discharge_kw'][k,ok];end[i,ok]=old['end_energy_kwh'][k,ok];reused[i,ok]=True
    for i,h in enumerate(hs):
        for j,day in enumerate(ds):
            if not inp[i,j] or reused[i,j]:continue
            if limit_pairs is not None and count>=limit_pairs:break
            nn=net[i,j]
            if scenarios is not None:
                rg,rp=scenarios;pathpv=np.maximum(forecast[i,j][None,:]+rp[i]*a['capacity_kwp'][h,day-1],0);pathpv[:,NIGHT]=0
                nn=np.maximum(load[i,j][None,:]+rg[i],0)-pathpv
            clock=time.perf_counter()
            try:
                cc,dd,ee,diag=qp.solve(nn,True);solve_times.append(time.perf_counter()-clock)
                c[i,j]=cc;d[i,j]=dd;end[i,j]=ee[-1]
                throughput[i,j]=battery.dt*np.sum(cc+dd)
                excl[i,j]=realized_bill(gc[i,j]-pv[i,j],cc,dd,buy,sell,battery)
                bill[i,j]=excl[i,j]+.09*np.sum(cl[i,j]);retail[i,j]=bill[i,j]-battery.throughput*throughput[i,j]
                residual=max(residual,diag['energy_balance_max']);maxsim=max(maxsim,diag['max_simultaneous_kw'])
            except Exception as exc:
                failures.append({'household':int(a['household_id'][h]),'date':str(a['date'][day]),'error':repr(exc)})
            count+=1
        if (i+1)%50==0:print(name,i+1,'/',len(hs),'seconds',round(time.perf_counter()-start,2),flush=True)
        if limit_pairs is not None and count>=limit_pairs:break
    # One accounting expression across cached and new actions. Cached float32
    # actions introduce <1e-6 AUD storage rounding, independently audited below.
    throughput=battery.dt*np.sum(c.astype(float)+d.astype(float),axis=-1)
    excl=realized_bill(gc-pv,c.astype(float),d.astype(float),buy,sell,battery)
    bill=excl+.09*np.sum(cl,axis=-1);retail=bill-battery.throughput*throughput
    if limit_pairs is not None:
        valid &= np.isfinite(end);single &= np.isfinite(end)
    if not failures:
        assert np.isfinite(excl[single]).all() and np.isfinite(bill[valid]).all()
    output=REV/'dispatch'/f'{name}.npz'
    np.savez_compressed(output,household_id=a['household_id'][hs],household_role=a['household_role'][hs],date=a['date'][ds],charge_kw=c,discharge_kw=d,bill_aud=bill,clean_day=valid,valid_gcgg_day=single,operating_excluding_cl_aud=excl,retail_bill_aud=retail,energy_throughput_kwh=throughput,start_energy_kwh=np.full(end.shape,5.),end_energy_kwh=end,actual_load_kw=gc+cl,actual_pv_kw=pv,forecast_load_kw=load,forecast_pv_kw=forecast,forecast_net_kw=net)
    report={'name':name,'tariff':kind,'seed':11,'households':len(hs),'dates':len(ds),'clean_feeder_household_days':int(valid.sum()),'clean_gcgg_household_days':int(single.sum()),'failures':failures,'battery':battery.__dict__,'max_energy_residual_kwh':residual,'max_simultaneous_kw':maxsim,'n_solved':len(solve_times),'cached_actions_count':int(reused.sum()),'cached_source':str(reuse_path),'solve_latency_seconds':{'median':float(np.median(solve_times)),'p95':float(np.quantile(solve_times,.95)),'mean':float(np.mean(solve_times))},'single_household_economic_estimand':'All valid history+GC/PV target days; controlled-load constant excluded if unavailable; absolute full cost uses feeder-valid individual days, not panel intersections.','output':str(output),'test_used_for_fitting_or_selection':False,**runtime_info(start)}
    write_json(output.with_suffix('.json'),report)
    if failures:raise RuntimeError(f'{name}: retained {len(failures)} failures')
    print(json.dumps(report),flush=True)
    return report
