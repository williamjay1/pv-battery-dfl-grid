"""Export the first already-fixed mechanism date; no effect-based selection."""
import json
from pathlib import Path
import numpy as np
from revision_network_physics import inputs
from network_replay import sha256
from revision_network_io import save_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_20261009/network_mechanism'

def main():
 plan=json.loads((OUT/'plan.json').read_text());chosen=plan['cases'][0]['date']
 ref=inputs('msecont');target=inputs('dfl');day=int(np.flatnonzero(ref['date']==chosen)[0])
 assert ref['clean_day'][day] and target['clean_day'][day]
 mechanism=np.load(OUT/'arrays.npz');arrays=dict(date=np.array(chosen),slot=np.arange(48),hour=np.arange(48)/2,
  household_id=ref['household_id'],house_phase=mechanism['house_phase'],house_bus=mechanism['house_bus'],
  actual_gross_load_kw=ref['actual_load_kw'][:,day].T,actual_pv_kw=ref['actual_pv_kw'][:,day].T)
 paths=[]
 for label,m,x in [('baseline','msecont',ref),('target','dfl',target)]:
  with np.load(x['path']) as d:
   hi=[int(np.flatnonzero(d['household_id']==h)[0]) for h in ref['household_id']]
   for key in ['forecast_pv_kw','forecast_load_kw','charge_kw','discharge_kw']:
    arrays[label+'_'+key]=d[key][hi,day].T
  arrays[label+'_battery_net_demand_kw']=arrays[label+'_charge_kw']-arrays[label+'_discharge_kw']
  arrays[label+'_net_demand_kw']=arrays['actual_gross_load_kw']-arrays['actual_pv_kw']+arrays[label+'_battery_net_demand_kw']
  arrays[label+'_net_injection_kw']=-arrays[label+'_net_demand_kw']
  path=ROOT/'results/network_replay/main_physical'/f'{m}_seed11_panel0_b55_all_tou_eulv_src1p05_fixed_test_physical.npz'
  with np.load(path) as d:
   arrays[label+'_voltage_pu']=d['household_voltage_pu'].reshape(365,48,55)[day].astype(float)
   arrays[label+'_source_import_kw']=d['source_import_kw'].reshape(365,48)[day]
  paths.extend([x['path'],path])
 for key in ['forecast_pv_kw','battery_net_demand_kw','net_demand_kw','net_injection_kw','voltage_pu','source_import_kw']:
  arrays['delta_'+key]=arrays['target_'+key]-arrays['baseline_'+key]
 np.savez_compressed(OUT/'first_prespecified_day.npz',**arrays)
 save_json(OUT/'first_prespecified_day.json',dict(date=chosen,selection='First date in previously frozen19-case plan, winter first clean date; not largest observed effect.',
  reference='continued MSE seed11',target='DFL seed11',panel=0,n_customers=55,n_slots=48,
  arrays='revision_20261009/network_mechanism/first_prespecified_day.npz',
  shapes={k:list(v.shape) for k,v in arrays.items()},units='Power kW, voltage pu, hour local dataset clock; all deltas target minus baseline.',
  signs='Net demand=GC+CL-PV+charge-discharge. Net injection is its negative. Forecast PV is decision signal, actual PV is shared fixed truth.',
  precision='Original stored AC voltage float32 converted to float64 for calculation, no additional precision claimed.',
  source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},code_sha256=sha256(__file__)))
 print(json.dumps(dict(status='completed',date=chosen,arrays=len(arrays))))
if __name__=='__main__':main()
