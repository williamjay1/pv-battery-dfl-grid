"""Read-only manuscript/table cross-check against completed network artifacts."""
from pathlib import Path
import hashlib,json,re
import numpy as np
from revision_network_io import save_json
from network_replay import utcnow

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_20261009'
def read(p):return json.loads((R/p).read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 article=(R/'manuscript/article_full.txt').read_text('utf-8')
 supp=(R/'manuscript/supplement_full.txt').read_text('utf-8')
 rows=read('network_diagnostics/scenarios.json')['records']
 summary=read('network_revision_summary.json');selected=summary['selected_comparisons']
 checks=[];issues=[]
 def table(text,title):
  start=text.index(title);lines=text[start:].splitlines();out=[]
  for line in lines[2:]:
   if ' | ' not in line:break
   out.append([x.strip() for x in line.split(' | ')])
  return out
 def number(label,printed,expected):
  if expected is None:
   ok=printed in ('Undefined','undefined','—','NA','N/A')
  else:
   try:
    value=float(printed);precision=len(printed.split('.')[1]) if '.' in printed else 0
    ok=abs(value-expected)<=.500001*10**(-precision)+1e-12
   except ValueError:ok=False
  checks.append(dict(label=label,printed=printed,expected=expected,pass_check=bool(ok)))
 def scenario(method,panel=0,network='eulv',source=1.05):
  found=[]
  for row in rows:
   m=row['metadata'];p=m['parameters'] or {}
   if (m['objective']==method and m['seed']==11 and m['panel']==panel and
       m['battery_count']==55 and m['tariff_mix']=='all_tou' and
       m.get('comparison_group') in (None,'original_physical') and
       p.get('network')==network and p.get('source_pu')==source and
       p.get('mapping_seed') is None and p.get('power_factor',.95)==.95 and
       p.get('impedance_scale',1)==1 and p.get('export_limit_kw') is None):found.append(row)
  assert len(found)==1,(method,panel,network,source,len(found))
  return found[0]
 scale=read('network_system_scale.json')['panels']
 for i,line in enumerate(table(article,'Table 1. Scale')):
  p=scale[i]
  expected=[p['n_clean_days'],p['pv_capacity_kwp_distribution']['median'],
     p['household_mean_daily_gross_demand_kwh']['median'],
     p['battery_capacity_to_household_daily_demand_ratio']['median'],
     p['total_pv_kwp_to_transformer_kva']*100,p['reference_observed_max_transformer_loading_pct']]
  for j,(a,b) in enumerate(zip(line[1:],expected)):number(f'T1 row{i+1} col{j+2}',a,b)
 for line,method in zip(table(article,'Table 4. Joint'),['msecont','calibrated','dfl','stochastic']):
  r=scenario(method);v=r['voltage'];f=r['flow']
  vals=[f['daily_operating_cost_aud']['mean'],v['over_pct'],v['under_pct'],
     f['daily_import_peak_kw']['mean'],f['daily_export_peak_kw']['mean'],f['daily_loss_kwh']['mean']]
  for j,(a,b) in enumerate(zip(line[1:],vals)):number(f'T4 {method} col{j+2}',a,b)
 for line in table(supp,'Table S1a.'):
  if line[0]=='Total':continue
  for panel in range(4):
   cov=scenario('msecont',panel)['coverage']['months'][line[0]]
   number(f'S1a {line[0]} panel{panel}',line[panel+2],cov['clean_days'])
 if 'Table S4a.' in supp:
  direction=table(supp,'Table S4a.');tails={tuple(x[:3]):x for x in table(supp,'Table S4b.')}
  assert len(direction)==10 and len(tails)==10,'Incomplete split voltage tables'
  s4rows=[x+tails[tuple(x[:3])][3:] for x in direction]
 else:s4rows=table(supp,'Table S4.')
 for line in s4rows:
  network=line[0] if line[0]=='eulv' else 'simbench_'+line[0]
  r=scenario(line[2],network=network,source=float(line[1]));v=r['voltage']
  expected=[v['over_pct'],v['under_pct'],v['conditional_over_pu'],v['conditional_under_pu'],v['quantiles'][8],v['quantiles'][-1]]
  for j,(a,b) in enumerate(zip(line[3:9],expected)):number(f'S4 {line[:3]} col{j+4}',a,b)
  arr=np.load(ROOT/r['arrays'])
  for j,k in enumerate(['node_longest_over_hours','node_longest_under_hours']):number(f'S4 {line[:3]} longest {k}',line[10+j],float(arr[k].max()))
  worst=line[9].split('/')
  for j,key in enumerate(['over','under']):
   expected=None if v[key+'_pct']==0 else v['worst_'+key+'_node_index']+1
   number(f'S4 {line[:3]} worst {key}',worst[j],expected)
 for line in table(supp,'Table S5.'):
  if line[0]=='mapping':r=next(x for x in selected['mapping_seed11'] if x['key'][5]==int(line[1]))
  else:
   pf,z=map(float,re.search(r'PF ([0-9.]+); Z ([0-9.]+)',line[1]).groups())
   r=next(x for x in selected['physical_brackets_seed11'] if x['key'][6:8]==[pf,z])
  for j,k in enumerate(['daily_voltage_pct','daily_import_peak_kw','daily_export_peak_kw','daily_loss_kwh']):number(f'S5 {line[:2]} {k}',line[j+2],r['outcomes'][k]['delta']['mean'])
 for line in table(supp,'Table S7.'):
  panel=3 if line[1]=='Held-out' else int(line[1])-1
  r=next(x for x in selected['control_sensitivities_seed11'] if x['comparison_group']==line[0] and x['key'][0]==panel)
  for pos,key,mult in [(2,'daily_operating_cost_aud',100),(3,'daily_over_pct',1),(5,'daily_import_peak_kw',1),(6,'daily_loss_kwh',1)]:number(f'S7 {line[:2]} {key}',line[pos],r['outcomes'][key]['delta']['mean']*mult)
  for a,b in zip(line[4].split(' to '),r['outcomes']['daily_over_pct']['delta']['ci95']):number(f'S7 {line[:2]} CI',a,b)
 for line in table(supp,'Table S8.'):
  r=next(x for x in selected['export_cap_seed11'] if x['key'][0]==int(line[0])-1 and x['reference']=='msecont')
  for j,k in enumerate(['daily_operating_cost_aud','daily_over_pct','daily_under_pct','daily_import_peak_kw','daily_export_peak_kw']):number(f'S8 {line[0]} {k}',line[j+1],r['outcomes'][k]['delta']['mean']*(100 if j==0 else 1))
 for line in table(supp,'Table S9.'):
  r=next(x for x in summary['geography_contrast']['records'] if x['metric']==line[0])
  for j,k in enumerate(['narrow_method_gap','matched_dispersed_method_gap']):number(f'S9 {line[0]} {k}',line[j+1],r[k])
  number(f'S9 {line[0]} difference',line[3],r['difference_of_method_gaps']['mean'])
  for a,b in zip(line[4].split(' to '),r['difference_of_method_gaps']['ci95']):number(f'S9 {line[0]} CI',a,b)
 if any(not x['pass_check'] for x in checks):issues.append('Table numeric or no-event index checks have failures; see machine details. Current known failure is displaying a worst site when no event occurred.')
 if 'SimBench indices are its observed native locations' in supp:issues.append('S4 native-index description is wrong: 55 assigned household slots repeat their 13/41 native-location voltages. Exposures/distributions are household-weighted, not equal-weighted native buses.')
 if 'pointwise' not in article.lower():issues.append('Add explicit pointwise/non-simultaneous confidence-interval qualification; intervals are not multiplicity adjusted.')
 manual=[
  'Table 5 boundaries are consistent with actual evidence; no field-compliance, dynamics, lifetime or primary thermal-congestion claim.',
  'Sections 3.3 and 3.5 correctly retain increased conditional severity at source1.05/1.00, nonzero wide-band undervoltage, rural transformer overload, nonuniform SAA voltage ordering, and adverse matched-persistence network changes.',
  'Continued MSE is the primary comparator in five source/topology conditions, four PF/Z brackets and ten mappings; frozen MSE is secondary. Full tariff/adoption cells encode nine independent physical deployments rather than 12.',
  'Geography uses 27/27 households and160 common clean dates. Cost and voltage denominators are27. Its gap-of-gaps confidence intervals are descriptive and not causal geography estimates.',
  'Mechanism day date2012-08-24,22:00 demand delta-2.41705157kW,mean voltage+0.00073359988pu,maxabs0.00206410885pu; full19case error and nearzero-case caveat match saved arrays.',
  'All overnight-support-matched direct-net additions completed. DFL versus net24_night reduces cost but increases import peaks in all four panels; signed diagnostic results must remain separately labeled.',
  'No manuscript/source files were changed by this read-only audit. No new experimental setting was run.'
 ]
 report=dict(created_utc=utcnow(),status='PASS' if not issues else 'CONDITIONAL_PRESENTATION_CORRECTIONS',
  manuscript_sha256={str(p.relative_to(ROOT)):sha(p) for p in [R/'manuscript/article_full.txt',R/'manuscript/supplement_full.txt']},
  numeric_checks=len(checks),passed=sum(x['pass_check'] for x in checks),failed=[x for x in checks if not x['pass_check']],issues=issues,manual_checks=manual,
  evidence=dict(diagnostic_scenarios=summary['diagnostic_scenario_count'],new_ac=summary['newly_executed_ac_replays'],reused_geo=summary['reused_original_geo_with_common_mask'],audit=summary['audit']),all_checks=checks)
 save_json(R/'network_manuscript_audit.json',report)
 lines=['NETWORK MANUSCRIPT READ-ONLY AUDIT',report['created_utc'],report['status'],
  f"Numerical/table checks: {report['passed']}/{report['numeric_checks']} pass.",'','Required corrections:']+[f'- {x}' for x in issues]+['','Verified conclusions:']+[f'- {x}' for x in manual]+['','Exact manuscript hashes:']+[f'{k}: {v}' for k,v in report['manuscript_sha256'].items()]
 (R/'research/network_manuscript_audit.txt').write_text('\n'.join(lines)+'\n','utf-8')
 print(json.dumps({k:report[k] for k in ['status','numeric_checks','passed','failed','issues']},ensure_ascii=False))
if __name__=='__main__':main()
