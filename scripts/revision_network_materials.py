"""Machine-readable revision evidence and brief, claim-bounded writing material."""
from __future__ import annotations
import json
from pathlib import Path
from network_replay import sha256,utcnow
from revision_network_io import save_json

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'revision_20261009'
def read(path):return json.loads((BASE/path).read_text('utf-8'))
def ordinary(r):
 k=r['key'];return (r.get('comparison_group','original_physical')=='original_physical' and k[3:]==['eulv',1.05,None,.95,1,None])
def compact(r):
 return {k:r[k] for k in ['key','reference','target','seeds','outcomes','n_customers','comparison_group'] if k in r}
def main():
 scenarios=read('network_diagnostics/scenarios.json')['records']
 pairs=read('network_diagnostics/paired.json')['records']
 fixed=read('network_diagnostics/fixed_seed11_paired.json')['records']
 audit=read('network_revision_audit.json')
 selected={
  'primary_three_seed_continued_reference':[compact(r) for r in pairs if ordinary(r) and r['reference']=='msecont' and r['key'][0] in [0,1,2]],
  'heldout_three_seed_continued_reference':[compact(r) for r in pairs if ordinary(r) and r['reference']=='msecont' and r['key'][0]==3],
  'five_conditions_seed11':[compact(r) for r in fixed if r.get('comparison_group','original_physical')=='original_physical' and r['reference']=='msecont' and r['key'][:3]==[0,55,'all_tou'] and r['key'][5:]==[None,.95,1,None]],
  'physical_brackets_seed11':[compact(r) for r in fixed if r.get('comparison_group','original_physical')=='original_physical' and r['reference']=='msecont' and r['key'][:3]==[0,55,'all_tou'] and r['key'][5] is None and r['key'][8] is None and (r['key'][6]!=.95 or r['key'][7]!=1)],
  'mapping_seed11':[compact(r) for r in fixed if r['reference']=='msecont' and r['key'][5] is not None],
  'all_practical_baselines_seed11':[compact(r) for r in fixed if ordinary(r) and r['key'][1:3]==[55,'all_tou'] and r['key'][0] in [0,1,2,3]],
  'export_cap_seed11':[compact(r) for r in fixed if r['key'][8] is not None],
  'control_sensitivities_seed11':[compact(r) for r in fixed if r.get('comparison_group','original_physical')!='original_physical'],
  'geography_seed11':[compact(r) for r in fixed if r['key'][0] in ['geo_narrow','geo_matched']]
 }
 conditional=[];curtail=[];coverage=[]
 for r in scenarios:
  m=r['metadata'];p=m['parameters'] or {}
  if m['panel']==0 and m['battery_count']==55 and m['tariff_mix']=='all_tou' and m['seed']==11 and m['objective'] in ['msecont','dfl'] and p.get('mapping_seed') is None and p.get('power_factor',.95)==.95 and p.get('impedance_scale',1)==1 and p.get('export_limit_kw') is None:
   conditional.append(dict(scenario_id=r['scenario_id'],network=p['network'],source_pu=p['source_pu'],method=m['objective'],voltage=r['voltage'],coverage=r['coverage']))
  if p.get('export_limit_kw') is not None:
   st=json.loads((ROOT/r['source']).with_suffix('.json').read_text())
   curtail.append(dict(panel=m['panel'],objective=m['objective'],n_clean_days=r['coverage']['clean_days'],
    mean_daily_curtailment_kwh_per_feeder=st['mean_daily_curtailment_kwh'],
    mean_lost_export_revenue_aud_per_house_day=st['mean_lost_export_revenue_aud_per_house_day']))
  if m['objective']=='msecont' and m['seed']==11 and m['panel'] in [0,1,2,3] and m['battery_count']==55 and m['tariff_mix']=='all_tou' and p.get('network')=='eulv' and p.get('source_pu')==1.05 and p.get('mapping_seed') is None and p.get('power_factor',.95)==.95 and p.get('impedance_scale',1)==1 and p.get('export_limit_kw') is None:
   coverage.append(dict(panel=m['panel'],**r['coverage']))
 pending=read('network_dispatch_batch_status.json').get('pending',[])
 result=dict(created_utc=utcnow(),status='completed' if not pending else 'Completed available-input evidence; controller groups explicitly pending',
  diagnostic_scenario_count=len(scenarios),ac_audited_new_scenario_count=audit['new_scenarios'],
  newly_executed_ac_replays=audit['newly_executed_ac_replays'],reused_original_geo_with_common_mask=audit['reused_original_geo_replays_with_new_common_mask'],
  primary_reference='continued MSE; frozen MSE secondary',selected_comparisons=selected,
  pooled_conditional_severity_five_conditions=conditional,export_curtailment=curtail,calendar_coverage=coverage,
  mechanism=read('network_mechanism/summary.json'),system_scale=read('network_system_scale.json'),
  geography_contrast=read('network_geography/paired_cohort_contrast.json'),
  pending_controller_groups=pending,
  definitions={
   'operating_cost':'General-meter retail settlement plus battery throughput wear plus separately metered controlled-load cost, AUD per household per retained day. Not synonymous with electricity bill.',
   'continuous_economics':'Network arrays report raw bill_aud operating cost. Inventory-adjusted economic estimates are separate dispatch analysis; do not substitute raw clean-day network means for segment-boundary-adjusted value.',
   'exposure':'Percent of occupied-customer half-hours on common strict source-clean dates. Over >1.05 and under <0.95 separately; fixed 1.10/0.90 sensitivity also stored.',
   'conditional_severity':'Pooled exceedance magnitude sum / corresponding pooled number of events. Null if no events. Daily conditional-ratio means are explicitly distinct and are not the pooled statistic.',
   'event_duration':'Consecutive observed half-hour slots, broken at missing dates. Longest duration=0 and episode count=0 with no events; episode p95 uses 0 as no-event sentinel and must be read with episode count. No sub-half-hour transient claim.',
   'percent_change':'(DFL mean minus reference mean) / reference mean times 100. Zero baseline gives null; percentage-point exposure differences are separate.',
   'uncertainty':'2000 synchronized circular seven-calendar-day bootstrap resamples over the complete 365-date axis, preserving missing-day positions. Three-seed mean formed before resampling for main/heldout; fixed seed11 throughout engineering conditions. Intervals are pointwise, not multiplicity-adjusted simultaneous coverage. Conditional on selected synthetic feeder deployment; no population/parameter confidence claim.',
   'independence':'12 primary tariff/adoption cells encode 9 distinct physical deployments; b28 fee-mix pair is physically duplicate. Neither slots nor seeds are independent sample replicates.',
   'physical_brackets':'Source 1.00/1.025/1.05; PF .90/.95/1.00; cable R/X multipliers .90/1.00/1.10 with unchanged transformer. One-factor deterministic assumptions, not measured uncertainty ranges.',
   'export_rule':'Prospective late-2026 Ausgrid new/upgraded connection scenario of 3 kW/site; not a historical 2012 constraint. Ideal aggregate-PCC PV-only curtailment, fixed battery actions/SOC, post-curtailment retail revenue recomputed. Not complete inverter-compliance simulation.',
   'geography':'27 eligible same-postcode customers versus 27 pretest demand/capacity and role matched customers from 21 other postcodes. Same first27 nodes and same160 clean dates. Not known real feeders or causal geographic effects.',
   'ratings':'Original EULV800 kVA, no thermal congestion claim; original transformer capacity and impedance preserved. Cable thermal ratings unspecified, no fabricated line-loading percentages.',
   'rounding':'Historical node voltages float32 have <=6e-8pu rounding near1pu; original float64 scalar violation metric retained separately. New simulations store float64.'},
  audit= {'status':audit['status'],'maximum_balance_residual_kw':audit['maximum_balance_residual_kw']},
  source_paths={'engineering':'revision_20261009/research/engineering_sources.json','raw_models':'research/network_provenance.json','strict_split':'datasets/splits_v1.json'},
  commands=[
   'python scripts/revision_network_replay.py --suite matched_external --workers 3',
   'python scripts/revision_network_replay.py --suite matched_mapping --workers 3',
   'python scripts/revision_network_physics.py --suite physical_brackets --workers 3',
   'python scripts/revision_network_physics.py --suite export_limit --panels 0 1 2 --workers 3',
   'python scripts/revision_network_geography.py',
   'python scripts/revision_network_mechanism.py',
   'python scripts/revision_network_mechanism_day.py',
   'python scripts/revision_network_dispatch_batch.py --workers 4',
   'python scripts/revision_network_diagnostics.py --scope all',
   'python scripts/revision_network_scale.py',
   'python scripts/revision_network_thermal.py',
   'python scripts/revision_network_finalize.py',
   'python scripts/revision_network_materials.py'],
  artifact_hashes={p:sha256(BASE/p) for p in ['network_diagnostics/scenarios.json','network_diagnostics/paired.json','network_diagnostics/fixed_seed11_paired.json','network_mechanism/arrays.npz','network_geography/cohort_plan.json']})
 save_json(BASE/'network_revision_summary.json',result)
 lines=['NETWORK REVISION EVIDENCE (machine table: network_revision_summary.json)',
  f"Diagnostic scenarios: {len(scenarios)}; newly audited artifacts: {audit['new_scenarios']}, comprising {audit['newly_executed_ac_replays']} newly executed AC replays and {audit['reused_original_geo_replays_with_new_common_mask']} original geo replays with newly intersected dates.",
  'Primary comparisons use continued MSE. Engineering condition figures use seed11 consistently.',
  'Reduced exceedance frequency does not imply lower conditional severity: at source1.05 pooled overvoltage magnitude increases slightly; at source1.00 pooled undervoltage magnitude also increases slightly.',
  'At source1.00 wide-band <0.90 undervoltage remains nonzero. SimBench zero narrow-band events imply undefined (null), not zero, conditional magnitude.',
  'Full55 panel0 baseline transformer loading peaks around25.3%; this is not a transformer-thermal-congestion experiment.',
  'July2012 and March2013 have zero jointly clean panel0 days. The 365-date calendar is not 365 observed clean days.',
  'ThreekW export limit is a prospective scenario. It is nearly inactive in panel0, materially curtails PV in panel1, and has intermediate exposure in panel2; report all three fixed panels.',
  'Local finite-difference sensitivities are in demand-positive kW/kvar. Full AC replay checks approximation error; they do not establish dynamic or real-feeder causal effects.',
  'Controller-change comparisons remain grouped by their actual common conditions. Raw operating costs in network tables do not replace continuous-SOC inventory-adjusted economics.',
  'All original study files remain intact; new material and calculations are stored on D. Raw source materials were not changed.']
 (BASE/'network_revision_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='completed',diagnostics=len(scenarios),tables={k:len(v) for k,v in selected.items()})),flush=True)
if __name__=='__main__':main()
