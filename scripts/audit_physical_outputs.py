"""Independent post-run loading, support, mask and scenario replay checks."""
from pathlib import Path
import json
import numpy as np
import torch
from forecast_models import load_data,build_masks,make_features
from train_decision_physical import ROOT,PhysicalPVMLP,NIGHT_SLOTS
from decision_baselines import residual_bank
from battery_model import DispatchQP

def main():
    torch.set_num_threads(2)
    p,splits=load_data();_,finite,_,_,masks=build_masks(p,splits)
    original=json.loads((ROOT/'results/decision_training_v1.json').read_text())
    pair=np.concatenate([np.argwhere(masks[k])[::997][:16] for k in ['train','validation','test','heldout_test']]);h,d=pair.T
    x=torch.from_numpy(make_features(p,pair));records=[]
    for family in ['decision','mse_continuation']:
        a=np.load(ROOT/f'datasets/{family}_arrays_physical.npz')
        assert np.array_equal(a['household_id'],p['household_id']) and np.array_equal(a['date'],p['date'])
        for kind in ['tou','flat']:
            r=json.loads((ROOT/f'results/{family}_training_physical_{kind}.json').read_text())
            assert r['status']=='complete'
            if family=='decision':
                assert r['train_pairs']==original['train_pairs'] and r['validation_pairs']==original['validation_pairs']
                assert not r['test_used_for_selection']
                selected=min(r['tuning'],key=lambda v:v['validation_best_bill'])['config']
                assert all(v['config']==selected for v in r['final'])
            else:
                assert r['same_train_and_validation_pairs_as_original_and_physical_dfl'] and not r['test_used_for_fitting_or_selection']
                selected=min(r['tuning'],key=lambda v:v['best_validation_bill'])['config'];assert selected==r['selected_config']
            for seed in [11,23,47]:
                key=f'pv_{"dfl" if family=="decision" else "msecont"}_{kind}_seed{seed}'
                y=a[key]
                assert np.isfinite(y[finite]).all() and (y[finite]>=0).all()
                assert (y[finite][:,NIGHT_SLOTS]==0).all()
                ck=torch.load(ROOT/f'checkpoints/{family}_physical_{kind}_seed{seed}.pt',weights_only=False)
                assert ck['model_class']=='PhysicalPVMLP' and ck['night_zero_slots']==NIGHT_SLOTS
                model=PhysicalPVMLP();model.load_state_dict(ck['model_state_dict']);model.eval()
                with torch.no_grad():expected=model(x).numpy()*p['capacity_kwp'][h,d-1,None]
                error=float(np.max(abs(expected-y[h,d])));assert error<1e-5
                record={'family':family,'tariff':kind,'seed':seed,'night_zero_exact':True,'max_reloaded_prediction_error_kw':error,'best_epoch':ck['log']['best_epoch'],'selected_lr':selected['learning_rate']}
                if family=='mse_continuation':
                    same=torch.load(ROOT/f'checkpoints/mse_same_lr_physical_{kind}_seed{seed}.pt',weights_only=False)
                    identity=all(torch.equal(v,same['model_state_dict'][k]) for k,v in ck['model_state_dict'].items())
                    if r['same_lr_equals_primary']:assert identity
                    record['strict_same_lr_equals_primary']=identity
                    record['primary_lr_equals_dfl_lr']=r['same_lr_equals_primary']
                records.append(record)
    b=np.load(ROOT/'datasets/forecast_arrays_v1.npz');baseline=[]
    for method in ['calibrated','stochastic']:
        for kind in ['tou','flat']:
            path=ROOT/f'results/dispatch/{method}_{kind}_seed11_test_physical.npz'
            meta=json.loads(path.with_suffix('.json').read_text());z=np.load(path)
            assert meta['failed']==0 and meta['max_forecast_or_scenario_night_pv_kw']==0
            hs=np.array([np.flatnonzero(b['household_id']==i)[0] for i in z['household_id']]);ds=np.flatnonzero(b['split_code']==2)
            expected=b['valid_history'][hs][:,ds]&b['valid_target_feeder'][hs][:,ds]
            assert np.array_equal(expected,z['clean_day']) and expected.sum()==79397
            for key in ['charge_kw','discharge_kw','bill_aud','end_energy_kwh','forecast_pv_kw']:assert np.isfinite(z[key][expected]).all()
            assert (z['forecast_pv_kw'][expected][:,NIGHT_SLOTS]==0).all()
            baseline.append({'method':method,'tariff':kind,'clean_household_days':int(expected.sum()),'failed':0,'exact_night_support':True,'max_energy_balance_residual':meta['max_energy_balance_residual']})
    # Rebuild the unchanged known-household bank and replay fixed known/held samples.
    keys=['household_id','household_role','date','split_code','valid_history','valid_target_gcgg','gc_shared','pv_mlp_seed11','capacity_kwp','truth_gc_kw','truth_pv_kw']
    bank={k:b[k] for k in keys};bank['pv_mlp_seed11']=bank['pv_mlp_seed11'].copy();ok=np.isfinite(bank['pv_mlp_seed11']).all(2)
    for slot in NIGHT_SLOTS:bank['pv_mlp_seed11'][:,:,slot][ok]=0
    rg,rp,bankmeta=residual_bank(bank,hs);rp[:,:,NIGHT_SLOTS]=0
    scenario_checks=[]
    for kind in ['tou','flat']:
        z=np.load(ROOT/f'results/dispatch/stochastic_{kind}_seed11_test_physical.npz');qp=DispatchQP(kind=kind,n_scenarios=20,tolerance=1e-13)
        for i in [0,165]:
            j=int(np.flatnonzero(z['clean_day'][i])[0]);day=ds[j];hh=hs[i]
            point=bank['pv_mlp_seed11'][hh,day];load=bank['gc_shared'][hh,day]
            pv=np.maximum(point[None,:]+rp[i]*bank['capacity_kwp'][hh,day],0);pv[:,NIGHT_SLOTS]=0
            net=np.maximum(load[None,:]+rg[i],0)-pv;c,d,e=qp.solve(net)
            error=max(float(np.max(abs(c-z['charge_kw'][i,j]))),float(np.max(abs(d-z['discharge_kw'][i,j]))));assert error<1e-5
            permutation=np.random.default_rng(100+i).permutation(20);cp,dp,_=qp.solve(net[permutation])
            pe=max(float(np.max(abs(c-cp))),float(np.max(abs(d-dp))));assert pe<1e-5
            scenario_checks.append({'tariff':kind,'household_id':int(z['household_id'][i]),'is_heldout':i==165,'date':str(z['date'][j]),'replay_max_action_difference_kw':error,'independent_household_scenario_permutation_difference_kw':pe})
    result={'status':'PASS','training_rows':len(original['train_pairs']),'validation_rows':len(original['validation_pairs']),
            'independent_code_review':'network_impl read-only PASS for factory, masks, pairs, budget and epoch0','models':records,'baselines':baseline,'scenario_replay':scenario_checks}
    (ROOT/'results/forecast_physical_output_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
