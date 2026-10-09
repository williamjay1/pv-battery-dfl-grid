"""System scale and prespecified household heterogeneity of revision results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from revision_network_io import save_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"revision_20261009"
def distribution(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return dict(n=int(len(x)),mean=float(x.mean()),min=float(x.min()),q10=float(np.quantile(x,.1)),
  median=float(np.median(x)),q90=float(np.quantile(x,.9)),max=float(x.max())) if len(x) else None
def main():
 with np.load(ROOT/"datasets/forecast_arrays_v1.npz") as z:
  ids=z["household_id"];dates=z["date"];split=z["split_code"];role=z["household_role"];cap=z["capacity_kwp"]
  gross=z["truth_gc_kw"]+z["truth_cl_kw"];pv=z["truth_pv_kw"];quality=z["valid_target_feeder"]
 test=split==2;pre=split!=2;idmap={int(h):i for i,h in enumerate(ids)}
 preenergy=np.nanmean(np.where(quality[:,pre],.5*gross[:,pre].sum(axis=2),np.nan),axis=1)
 precap=np.nanmedian(cap[:,pre],axis=1)
 boundaries=dict(capacity_kwp=np.nanquantile(precap[role==0],[1/3,2/3]).tolist(),
                 pretest_daily_demand_kwh=np.nanquantile(preenergy[role==0],[1/3,2/3]).tolist())
 splits=json.loads((ROOT/"datasets/splits_v1.json").read_text())
 rows=[]
 for panel,hs in enumerate(splits["primary_panels"]+[splits["heldout_network_household_ids"]]):
  hi=np.array([idmap[int(h)] for h in hs])
  ref=ROOT/"results/network_replay"/("main_physical" if panel<3 else "heldout_physical")/f"msecont_seed11_panel{panel}_b55_all_tou_eulv_src1p05_fixed_test_physical.npz"
  with np.load(ref) as n:
   mask=n["clean_day"];baseline_loading=n["transformer_loading_pu"].reshape(-1,48)[mask]
   baseline_phase=n["transformer_max_phase_loading_pu"].reshape(-1,48)[mask]
   imports=n["source_import_kw"].reshape(-1,48)[mask]
  loads=.5*gross[hi][:,test][:,mask].sum(axis=2);solar=.5*pv[hi][:,test][:,mask].sum(axis=2)
  capacity=np.median(cap[hi][:,test][:,mask],axis=1)
  nofile=next((ROOT/"results/network_replay/none_physical").glob(f"none_seed11_panel{panel}_*.npz"),None)
  no={}
  if nofile:
   with np.load(nofile) as n:
    ll=n["transformer_loading_pu"].reshape(-1,48)[mask]
    no=dict(no_battery_max_transformer_loading_pct=float(ll.max()*100),no_battery_mean_daily_peak_transformer_loading_pct=float(ll.max(axis=1).mean()*100))
  rows.append(dict(panel=panel,n_households=len(hs),n_clean_days=int(mask.sum()),transformer_kva=800,
    total_pv_kwp=float(capacity.sum()),total_pv_kwp_to_transformer_kva=float(capacity.sum()/800),
    pv_capacity_kwp_distribution=distribution(capacity),household_mean_daily_gross_demand_kwh=distribution(loads.mean(axis=1)),
    household_mean_daily_pv_energy_kwh=distribution(solar.mean(axis=1)),
    full_adoption_battery_capacity_kwh=550,full_adoption_battery_power_kw=165,
    battery_capacity_to_household_daily_demand_ratio=distribution(10/loads.mean(axis=1)),
    full_adoption_battery_power_to_transformer_kva=165/800,
    reference_mean_daily_import_peak_kw=float(imports.max(axis=1).mean()),reference_observed_max_import_kw=float(imports.max()),
    reference_mean_daily_peak_transformer_loading_pct=float(baseline_loading.max(axis=1).mean()*100),
    reference_observed_max_transformer_loading_pct=float(baseline_loading.max()*100),
    reference_observed_max_phase_transformer_loading_pct=float(baseline_phase.max()*100),**no))
 engineering=[]
 for r in json.loads((OUT/'network_diagnostics/scenarios.json').read_text())['records']:
  m=r['metadata'];p=m.get('parameters') or {};meta=m.get('network_metadata') or {}
  if not (m['objective']=='msecont' and m['seed']==11 and m['panel']==0 and m['battery_count']==55 and m['tariff_mix']=='all_tou'):continue
  if p.get('mapping_seed') is not None or p.get('power_factor',.95)!=.95 or p.get('impedance_scale',1)!=1 or p.get('export_limit_kw') is not None:continue
  rating=float(meta.get('transformer_kva',sum(meta.get('transformer_ratings_kva',[]))))
  with np.load(ROOT/r['source']) as n:
   mask=n['clean_day'];loading=n['transformer_loading_pu'].reshape(-1,48)[mask]
   line=n.get('line_loading_pu',None)
   line_summary={}
   if line is not None and np.isfinite(line).any():
    l=line.reshape(-1,48)[mask]
    line_summary=dict(max_line_loading_pct=float(np.nanmax(l)*100),fraction_halfhours_any_line_above_rating=float(np.mean(l>1)))
  engineering.append(dict(network=p['network'],source_pu=p['source_pu'],original_transformer_kva=rating,
   n_customers=55,n_load_locations=meta.get('native_load_locations',55),balanced_only=meta.get('balanced_only',False),
   total_pv_kwp=rows[0]['total_pv_kwp'],pv_kwp_to_transformer_kva=rows[0]['total_pv_kwp']/rating,
   mean_gross_energy_kwh_per_feeder_day=rows[0]['household_mean_daily_gross_demand_kwh']['mean']*55,
   mean_pv_energy_kwh_per_feeder_day=rows[0]['household_mean_daily_pv_energy_kwh']['mean']*55,
   max_transformer_loading_pct=float(loading.max()*100),mean_daily_peak_transformer_loading_pct=float(loading.max(axis=1).mean()*100),
   fraction_halfhours_transformer_above_rating=float(np.mean(loading>1)),**line_summary))
 save_json(OUT/"network_system_scale.json",dict(reference="continued MSE seed11, full55 ToU adoption",primary_transformer="Original800kVA unchanged",
   quantile_group_cutpoints_from_known_pretest=boundaries,panels=rows,
   engineering_conditions=engineering,
   temporal_boundary="Observed clean half-hourly quasi-static samples, not subinterval thermal or voltage extrema. Historical nameplate PV power is median reported capacity on common clean test dates."))
 path=OUT/"network_diagnostics/paired.json"
 if path.exists():
  pairs=json.loads(path.read_text())["records"];hetero=[]
  for pair in pairs:
   if pair.get('comparison_group','original_physical')!='original_physical':continue
   if pair["reference"]!="msecont" or pair["key"][0] not in [0,1,2,3]:continue
   if pair["key"][3]!="eulv" or pair["key"][4]!=1.05 or pair["key"][5] is not None or pair["key"][6:]!=[.95,1,None]:continue
   r=pair["households"];hs=np.array(r["household_id"]);ii=np.array([idmap[int(h)] for h in hs])
   cost=np.array(r["cost_delta_aud_per_day"]);vhs=r["voltage_household_id"];v=np.array(r["voltage_delta_pp"])
   vlookup={int(h):vv for h,vv in zip(vhs,v)};vv=np.array([vlookup[int(h)] for h in hs])
   adoption=pair["key"][1];owner=np.arange(len(hs))<adoption
   groups=[]
   for variable,values in [("capacity_kwp",precap[ii]),("pretest_daily_demand_kwh",preenergy[ii])]:
    group=np.digitize(values,boundaries[variable],right=True)
    for g in range(3):
     mask=(group==g);o=mask&owner
     groups.append(dict(variable=variable,tertile=g+1,n_households=int(mask.sum()),n_owners=int(o.sum()),
       owner_cost_delta_aud_per_day=distribution(cost[o]),voltage_delta_pp=distribution(vv[mask]),
       owner_benefiting_fraction=float(np.mean(cost[o]<-1e-8)) if o.any() else None,
       owner_harmed_fraction=float(np.mean(cost[o]>1e-8)) if o.any() else None))
   hetero.append(dict(key=pair["key"],seeds=pair["seeds"],groups=groups))
  save_json(OUT/"network_household_groups.json",dict(cutpoints=boundaries,records=hetero,
    boundary="Descriptive prespecified capacity/pretest-use tertiles; no selected-significant subgroup claim; households share constructed-feeder exposure."))
 print(json.dumps({"status":"completed","panels":len(rows)}))
if __name__=="__main__":main()
