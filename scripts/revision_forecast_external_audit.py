"""Post-run external audit; never backdates the original model freeze."""
from revision_forecast_common import *
from datetime import datetime,timezone
import hashlib

def main():
    start=time.perf_counter();freeze=json.loads((REV/'research/external_model_freeze.json').read_text());original={v['relative_path']:v for v in freeze['artifacts']};weights=[]
    for fee in ['tou','flat']:
        for method in ['mse_continuation','decision']:
            for seed in [11,23,47]:
                name=f'checkpoints/{method}_physical_{fee}_seed{seed}.pt';p=ROOT/name;sha=hashlib.sha256(p.read_bytes()).hexdigest();weights.append({'relative_path':name,'sha256':sha,'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),'included_in_original_freeze':name in original,'matches_original_hash':sha==original[name]['sha256'] if name in original else None})
    checks=[];maxcost=maxenergy=maxend=maxnight=0.
    for label in ['2p2','2p0','2p4']:
        p=dict(np.load(REV/f'cache/storenet_external_reference_{label}kwp.npz'));mask=p['external_test_mask'];pairs=np.argwhere(mask);h,d=pairs.T
        assert len(pairs)==2224 and p['valid_history'][mask].all() and p['valid_gc_gg_day'][mask].all()
        for fee in ['tou','flat']:
            buy,sell=tariff(fee)
            for method in ['msecont','dfl']:
                for seed in [11,23,47]:
                    name=f'storenet_{label}_{method}_{fee}_seed{seed}';z=np.load(REV/f'dispatch/{name}.npz');assert np.array_equal(z['pairs'],pairs) and np.array_equal(z['clean_day'],mask);c=z['charge_kw'];dd=z['discharge_kw'];actual=p['power_kw'][h,d,:,0]-p['power_kw'][h,d,:,2]
                    assert np.array_equal(z['actual_load_kw'],p['power_kw'][h,d,:,0]) and np.array_equal(z['actual_pv_kw'],p['power_kw'][h,d,:,2]);bill=realized_bill(actual,c,dd,buy,sell);maxcost=max(maxcost,float(abs(bill-z['bill_aud'][mask]).max()));end=5.+np.cumsum(.5*(.95*c-dd/.95),axis=-1);maxenergy=max(maxenergy,float(max(0.,1-end.min(),end.max()-9)));maxend=max(maxend,float(abs(end[:,-1]-5).max()));maxnight=max(maxnight,float(abs(z['forecast_pv_kw'][:,NIGHT]).max()));checks.append(name)
    report={'status':'PASS' if maxcost<1e-8 and maxenergy<1e-5 and maxend<1e-5 and maxnight==0 else 'FAIL','recorded_utc':datetime.now(timezone.utc).isoformat(),'timing_disclosure':'This is a post-run audit. Six flat-tariff weight files were not listed in the original pre-run freeze; their current hashes/mtimes are disclosed here without backdating. The primary ToU model files were in the pre-run freeze.','loaded_model_weights':weights,'checked_dispatch_files':checks,'max_cost_recomputation_error_aud':maxcost,'max_storage_rounded_soc_bound_violation_kwh':maxenergy,'max_storage_rounded_cyclic_end_error_kwh':maxend,'max_night_signal_kw':maxnight,'mask_pairs_per_scale':2224,'uncertainty_scope':'paired calendar-block uncertainty conditional on these 10 households, not population household-sampling uncertainty',**runtime_info(start)};write_json(REV/'results/storenet_external_postrun_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['loaded_model_weights','checked_dispatch_files']}));assert report['status']=='PASS'

if __name__=='__main__':main()
