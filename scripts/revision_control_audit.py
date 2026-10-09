"""Audit all completed standard revision dispatch artifacts, without model edits."""
from revision_forecast_common import *

def main():
    initialize();start=time.perf_counter();records=[]
    for path in sorted((REV/'dispatch').glob('*.npz')):
        if path.name.startswith('storenet_') or 'pilot' in path.name:continue
        jp=path.with_suffix('.json')
        if not jp.exists():continue
        meta=json.loads(jp.read_text());z=np.load(path);kind='flat' if '_flat' in path.stem else 'tou';wear=meta.get('battery',{}).get('throughput',.02);b=Battery(throughput=wear);buy,sell=tariff(kind);c=z['charge_kw'].astype(float);d=z['discharge_kw'].astype(float);hh=np.asarray(z['household_id']);dates=z['date'];mask=z['valid_gcgg_day'];full=z['clean_day'];soc=z['start_energy_kwh'][:,:,None]+np.cumsum(.5*(.95*c-d/.95),axis=-1);finite=np.isfinite(c).all(-1)&np.isfinite(d).all(-1)
        # Full actual_load includes CL; use source measurements to recompute
        # both per-channel denominators and retail accounting independently.
        source=np.load(ROOT/'datasets/forecast_arrays_v1.npz');hi=np.asarray([np.flatnonzero(source['household_id']==v)[0] for v in hh]);di=np.asarray([np.flatnonzero(source['date']==v)[0] for v in dates]);gc=source['truth_gc_kw'][hi][:,di];pv=source['truth_pv_kw'][hi][:,di];cl=source['truth_cl_kw'][hi][:,di];expected=source['valid_history'][hi][:,di]&source['valid_target_gcgg'][hi][:,di];expectedfull=source['valid_history'][hi][:,di]&source['valid_target_feeder'][hi][:,di];assert np.array_equal(mask,expected) and np.array_equal(full,expectedfull);assert finite[mask].all()
        cost=realized_bill(gc-pv,c,d,buy,sell,b);bill=cost+.09*cl.sum(-1);throughput=.5*(c+d).sum(-1);err=float(abs(cost[mask]-z['operating_excluding_cl_aud'][mask]).max());berr=float(abs(bill[full]-z['bill_aud'][full]).max());bounds=float(max(0.,1-np.nanmin(soc),np.nanmax(soc)-9,-np.nanmin(c),-np.nanmin(d),np.nanmax(c)-3,np.nanmax(d)-3));enderr=float(np.nanmax(abs(soc[:,:,-1]-z['end_energy_kwh'])));sim=float(np.nanmax(np.minimum(c,d)));continuity=None;segments=None
        if 'segment_id' in z.files:
            sid=z['segment_id'];same=(sid[:,1:]==sid[:,:-1])&(sid[:,1:]>=0);continuity=float(abs(z['start_energy_kwh'][:,1:]-z['end_energy_kwh'][:,:-1])[same].max());starts=mask&~np.c_[np.zeros(len(mask),bool),(sid[:,1:]==sid[:,:-1])&(sid[:,1:]>=0)];assert np.max(abs(z['start_energy_kwh'][starts]-5))<1e-10;segments=int(starts.sum());assert continuity<1e-10
        else:assert np.max(abs(z['end_energy_kwh'][mask]-5))<1e-5
        passed=err<3e-6 and berr<3e-6 and bounds<1e-5 and enderr<1e-5 and sim<1e-5
        records.append({'file':path.name,'status':'PASS' if passed else 'FAIL','gcgg_days':int(mask.sum()),'full_cost_days':int(full.sum()),'cost_recompute_max_aud':err,'full_cost_recompute_max_aud':berr,'stored_float32_bound_violation':bounds,'stored_float32_end_recursion_error_kwh':enderr,'max_simultaneous_kw':sim,'continuous_soc_gap_kwh':continuity,'segments':segments});print(path.name,records[-1]['status'],flush=True)
    report={'status':'PASS' if all(r['status']=='PASS' for r in records) else 'FAIL','n_completed_artifacts':len(records),'scope':'All completed standard revision dispatch files present at audit time; StoreNet audited separately; incomplete files skipped until their JSON completion marker exists.','note':'Tolerance here accounts for stored float32 charge/discharge; raw solver feasibility diagnostics are separately retained in each dispatch JSON.','records':records,**runtime_info(start)};write_json(REV/'results/revision_dispatch_audit.json',report);assert report['status']=='PASS'

if __name__=='__main__':main()
