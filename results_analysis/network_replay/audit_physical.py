"""Post-run integrity checks of all fixed physical-support replay scenarios.

This audit reads outcomes only to check arithmetic, pairing, and completeness.
It does not select scenarios, thresholds, households, or outcomes.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from network_replay import write_manifest, save_json, sha256

EXPECTED = {'main_physical': 108, 'none_physical': 3,
            'heldout_physical': 18, 'external_full_physical': 8,
            'mapping_full_physical': 20, 'strong_baselines_physical': 6}


def network_audit():
    records, pairs, counts, hashes, support_checks = [], defaultdict(list), Counter(), {}, {}
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text(encoding='utf-8'))
    for folder, expected in EXPECTED.items():
        for path in sorted((ROOT / 'results/network_replay' / folder).glob('*.npz')):
            meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path) as z:
                a = {k: z[k] for k in z.files}
            D, H = len(a['date']), len(a['household_id'])
            p = meta['parameters']
            frozen_clean = (splits['primary_panel_common_clean_dates'][meta['panel']]['test']
                            if meta['panel'] < 3 else splits['heldout_network_common_clean_dates']['test'])
            frozen_ids = (splits['primary_panels'][meta['panel']] if meta['panel'] < 3
                          else splits['heldout_network_household_ids'])
            checks = {
                'completed': meta['status'] == 'completed',
                '365_dates_55_customers': D == 365 and H == 55,
                'household_order_matches_frozen_protocol': np.array_equal(a['household_id'],np.asarray(frozen_ids)),
                'dimensions': a['household_voltage_pu'].shape == (D*48,H) and a['bill_byhouse_aud'].shape == (D,H),
                'clean_subset_physical': bool(np.all(~a['clean_day'] | a['converged_day'])),
                'clean_dates_match_frozen_protocol': np.array_equal(a['date'][a['clean_day']],np.asarray(frozen_clean)),
                'valid_converged': np.array_equal(a['input_valid'],a['converged']),
                'voltage_nan_missing': bool(np.isnan(a['household_voltage_pu'][~a['converged']]).all()),
                'voltage_finite_valid': bool(np.isfinite(a['household_voltage_pu'][a['converged']]).all()),
                'mean_bill': np.allclose(a['bill_byhouse_aud'].mean(axis=1),a['bill_mean_aud'],atol=0,rtol=0,equal_nan=True),
                'house_id_mapping': np.array_equal(a['voltage_household_id'],a['household_id'][a['mapping_household_index']]),
                'sha256': sha256(path) == meta['output_sha256'],
                'physical_suffix': p['dispatch_suffix'] == 'physical',
                'tight_tolerance': all(s['report']['solver_tolerance'] == 1e-13 for s in meta['sources']),
                'no_test_fitting': all(s['report']['test_used_for_fitting'] is False for s in meta['sources']),
                'zero_ac_solver_failures': meta['solver_failed_slots'] == 0,
                'unmodified_network_code': sha256(ROOT/'scripts/network_model.py') == meta['network_model_sha256']}
            for source in meta['sources']:
                key = source['file']
                if key not in hashes:
                    hashes[key] = sha256(key)
                    if source['report'].get('model',source['report'].get('method')) != 'none':
                        with np.load(key) as dispatch:
                            night=dispatch['forecast_pv_kw'][...,np.r_[0:8,42:48]]
                            support_checks[key]=bool(np.all(night[np.isfinite(night)]==0))
                checks['all_input_hashes'] = checks.get('all_input_hashes',True) and hashes[key] == source['sha256']
                checks['nighttime_pv_exact_zero'] = checks.get('nighttime_pv_exact_zero',True) and support_checks.get(key,True)
            balance = float(np.nanmax(np.abs(a['power_balance_residual_kw'])))
            checks['power_balance_below_1e_minus4_kw'] = balance < 1e-4
            records.append({'file':str(path.relative_to(ROOT)), 'passed':all(checks.values()),
                            'checks':{k:bool(v) for k,v in checks.items()},
                            'max_power_balance_residual_kw':balance})
            counts[folder] += 1
            group = (folder,meta['seed'],meta['panel'],meta['battery_count'],meta['tariff_mix'],p['network'],p['source_pu'],p['mapping_seed'])
            pairs[group].append((meta['objective'],{k:a[k] for k in ['household_id','date','clean_day','input_valid','mapping_household_index']}))
    pair_checks=[]
    group_methods={'main_physical':{'mlp','dfl','msecont'},'heldout_physical':{'mlp','dfl','msecont'},
                   'external_full_physical':{'mlp','dfl'},'mapping_full_physical':{'mlp','dfl'},
                   'strong_baselines_physical':{'hgb','calibrated','stochastic'},'none_physical':{'none'}}
    methods_complete=all(set(e[0] for e in entries)==group_methods[group[0]] for group,entries in pairs.items())
    for group, entries in pairs.items():
        if len(entries)<2:
            continue
        checks = {key:all(np.array_equal(entries[0][1][key],e[1][key]) for e in entries[1:]) for key in entries[0][1]}
        pair_checks.append({'group':list(group),'methods':[x[0] for x in entries],
                            'checks':checks,'passed':all(checks.values())})
    inventory_ok = dict(counts) == EXPECTED and methods_complete
    value={'created_utc':datetime.now(timezone.utc).isoformat(),
           'n_scenarios':len(records),'expected_counts':EXPECTED,'actual_counts':dict(counts),
           'inventory_passed':inventory_ok,'n_paired_groups':len(pair_checks),
           'all_passed':inventory_ok and all(r['passed'] for r in records+pair_checks),
           'scope':'Source hashes, uniform physical-support suffix and solver tolerance, 365-date axis, common source-defined clean masks, customer ordering, billing denominators, convergence, power balance, output hashes and cross-method pairing; no selection on effects.',
           'scenarios':records,'paired_groups':pair_checks,
           'max_power_balance_residual_kw':max(r['max_power_balance_residual_kw'] for r in records)}
    save_json(ROOT/'results/network_replay/integrity_physical_complete.json',value)
    write_manifest()
    return value


def geography_audit():
    folder=ROOT/'results/geographic_replay/geographic_physical'
    records=[]
    reference=None
    postcodes=json.loads((ROOT/'datasets/household_postcodes.json').read_text(encoding='utf-8'))['by_household_id']
    excluded=json.loads((ROOT/'datasets/splits_v1.json').read_text(encoding='utf-8'))['excluded_ids']
    expected_ids=np.asarray(sorted(int(h) for h,p in postcodes.items() if p=='2259' and int(h) not in excluded))
    with np.load(ROOT/'datasets/forecast_arrays_v1.npz') as source:
        hs=np.asarray([int(np.flatnonzero(source['household_id']==h)[0]) for h in expected_ids])
        ds=np.flatnonzero(source['split_code']==2)
        source_dates=source['date'][ds]
        source_clean=(source['valid_history'][hs][:,ds]&source['valid_target_feeder'][hs][:,ds]).all(axis=0)
    for path in sorted(folder.glob('*_test_physical.npz')):
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(path) as z:
            a={k:z[k] for k in z.files}
        H=len(a['household_id']); D=len(a['date'])
        checks={
            '27_actual_ids':H==27 and 2 not in a['household_id'],
            'metadata_selected_eligible_ids':np.array_equal(a['household_id'],expected_ids),
            'source_defined_date_and_clean_mask':np.array_equal(a['date'],source_dates) and np.array_equal(a['clean_day'],source_clean),
            '27_customer_voltages':a['household_voltage_pu'].shape==(365*48,27),
            '55_load_points':a['all_load_point_voltage_pu'].shape==(365*48,55),
            'occupancy':np.array_equal(a['occupied_load_point'],np.r_[np.ones(27,dtype=bool),np.zeros(28,dtype=bool)]),
            'occupied_voltage_columns':np.array_equal(a['household_voltage_pu'],a['all_load_point_voltage_pu'][:,:27]),
            'original_transformer':meta['network_metadata']['transformer_kva']==800,
            'exceedance_denominator_27':np.allclose(a['voltage_exceedance_mean_pu'],a['voltage_exceedance_pu']/27,rtol=0,atol=0,equal_nan=True),
            'bill_denominator_27':np.array_equal(a['bill_byhouse_aud'].mean(axis=1),a['bill_mean_aud']),
            'clean340':int(a['clean_day'].sum())==340,
            'full_physical_convergence':D==365 and bool(a['converged'].all()),
            'file_hash':sha256(path)==meta['output_sha256'],
            'independent_dispatch_overlap':meta['formal_overlap_audit']['status']=='passed' and meta['formal_overlap_audit']['max_action_difference_kw']==0.0,
            'paired_fields':True}
        if reference is not None:
            checks['paired_fields']=all(np.array_equal(reference[k],a[k]) for k in ['date','household_id','clean_day','converged_day','occupied_load_point'])
        reference=a
        records.append({'scenario':meta['scenario'],'checks':{k:bool(v) for k,v in checks.items()},
                        'passed':all(checks.values()),'formal_overlap_audit':meta['formal_overlap_audit'],
                        'max_power_balance_residual_kw':float(np.nanmax(np.abs(a['power_balance_residual_kw'])))})
    value={'n_scenarios':len(records),'all_passed':len(records)==3 and all(r['passed'] for r in records),'scenarios':records}
    save_json(folder/'integrity_audit.json',value)
    return value


if __name__=='__main__':
    if '--geography-only' in sys.argv:
        result={'geography':geography_audit()}
    else:
        result={'network':network_audit(),'geography':geography_audit()}
    print(json.dumps({key:{k:v for k,v in value.items() if k in ['n_scenarios','n_paired_groups','all_passed','actual_counts','max_power_balance_residual_kw']} for key,value in result.items()}))
    if not all(value['all_passed'] for value in result.values()):
        raise SystemExit(1)
