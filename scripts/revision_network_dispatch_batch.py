"""Replay explicit new-controller pairs without mixing operating conditions.

Original fixed daily baselines use all four panels. Controller and wear changes
use fixed panel 0 and held-out panel 3, as agreed before examining their effects.
Missing inputs remain pending in the manifest, never silently removed.
"""
from __future__ import annotations
import argparse, concurrent.futures, json
from pathlib import Path
from revision_network_physics import run_case, inputs, BASE
from network_replay import utcnow
from revision_network_io import save_json
import numpy as np

def definitions():
 groups=[]
 for name in ['seasonal12','net_mse','net_calibrated24','net_mse_night','net_calibrated24_night']:
  groups.append(dict(group='original_physical',panels=[0,1,2,3],methods={name:name+'_tou_seed11_expanded'},singleton=True))
 # Original panel0 already has these same fixed controllers; use new expanded
 # dispatches to extend the evidence to the remaining fixed constructed panels.
 for name,stem in [('calibrated','calibrated3'),('stochastic','stochastic20')]:
  groups.append(dict(group='original_physical',panels=[1,2,3],methods={name:stem+'_tou_seed11_expanded'},singleton=True))
 groups += [
  dict(group='continuous_frozen_transfer',panels=[0,3],methods={'msecont':'continuous_frozen_msecont_tou','dfl':'continuous_frozen_dfl_tou'}),
  dict(group='continuous_matched_retraining',panels=[0,3],methods={'msecont':'continuous_matched_mse_tou_expanded','dfl':'continuous_matched_dfl_tou_expanded'})]
 for load in ['persistence','strong']:
  groups += [
   dict(group='load_'+load+'_frozen_transfer',panels=[0,3],methods={m:'frozen_'+m+'_load'+load+'_tou' for m in ['msecont','dfl']}),
   dict(group='load_'+load+'_matched_retraining',panels=[0,3],methods={m:'matched_pv_'+load+'_'+suffix+'_tou_wear002_expanded' for m,suffix in [('msecont','mse'),('dfl','dfl')]})]
 for wear in ['0','0.01','0.04']:
  groups += [
   dict(group='wear'+wear+'_frozen_transfer',panels=[0,3],methods={m:'frozen_'+m+'_tou_wear'+wear for m in ['msecont','dfl']}),
   dict(group='wear'+wear+'_matched_retraining',panels=[0,3],methods={m:'matched_pv_hgb_'+suffix+'_tou_wear'+wear+'_expanded' for m,suffix in [('msecont','mse'),('dfl','dfl')]})]
 return groups

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
 parser.add_argument('--groups',nargs='*');parser.add_argument('--methods',nargs='*');args=parser.parse_args()
 tasks=[];pending=[];selected=[]
 for group in definitions():
  if args.groups and group['group'] not in args.groups:continue
  if args.methods and not any(m in args.methods for m in group['methods']):continue
  paths={m:BASE/'dispatch'/(stem+'.npz') for m,stem in group['methods'].items()}
  missing=[str(p) for p in paths.values() if not p.exists() or not p.with_suffix('.json').exists()]
  if missing:pending.append(dict(**group,missing=missing));continue
  for panel in group['panels']:
   loaded=[inputs(m,panel=panel,dispatch_path=str(p)) for m,p in paths.items()]
   for x in loaded[1:]:
    assert np.array_equal(loaded[0]['date'],x['date'])
    assert np.array_equal(loaded[0]['household_id'],x['household_id'])
    assert np.array_equal(loaded[0]['clean_day'],x['clean_day']),group['group']
   for m,path in paths.items():
    label=m if group['group']=='original_physical' else group['group']+'_'+m
    tasks.append(dict(suite='new_dispatch',objective=label,base_objective=m,
       comparison_group=group['group'],panel=panel,dispatch_path=str(path)))
  selected.append(group)
 manifest=dict(started_utc=utcnow(),status='running',groups=selected,pending=pending,tasks=tasks,results=[])
 status_suffix='_'+'_'.join(args.groups) if args.groups else ''
 if args.methods:status_suffix+='_methods_'+'_'.join(args.methods)
 dest=BASE/('network_dispatch_batch_status'+status_suffix+'.json');save_json(dest,manifest)
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(run_case,t):t for t in tasks}
  for fut in concurrent.futures.as_completed(futures):
   try:manifest['results'].append(fut.result())
   except Exception as exc:manifest['results'].append(dict(status='failed',task=futures[fut],error=str(exc)))
   save_json(dest,manifest)
 manifest['status']='completed_available_inputs' if all(r['status']=='completed' for r in manifest['results']) else 'failed'
 manifest['finished_utc']=utcnow();save_json(dest,manifest)
 print(json.dumps(dict(status=manifest['status'],scenarios=len(tasks),pending_groups=len(pending))),flush=True)
if __name__=='__main__':main()
