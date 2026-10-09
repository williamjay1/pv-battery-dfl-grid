"""Equal-size geographic contrast: matching uses pretest load/PV and role only."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,networkx as nx
from geographic_replay import read_inputs
from battery_model import Battery,DispatchQP,tariff,realized_bill
from network_model import EuropeanLV
from network_replay import save_json,sha256,utcnow
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"revision_20261009/network_geography"
REPLAY=ROOT/"revision_20261009/network_replay/geo_comparison"
def choose():
 post=json.loads((ROOT/"datasets/household_postcodes.json").read_text())["by_household_id"]
 with np.load(ROOT/"datasets/forecast_arrays_v1.npz") as a:
  ids=a["household_id"];role=a["household_role"];pre=a["split_code"]!=2
  eligible=(role!=2)
  gc=a["truth_gc_kw"][:,pre];cl=a["truth_cl_kw"][:,pre];valid=a["valid_target_feeder"][:,pre]
  energy=np.where(valid,.5*(gc+cl).sum(axis=2),np.nan)
  load=np.nanmean(energy,axis=1);capacity=np.nanmedian(a["capacity_kwp"][:,pre],axis=1)
  availability=valid.mean(axis=1)
 geo=np.array(sorted(int(h) for h in ids[eligible] if post[str(h)]=="2259"))
 candidates=np.array([int(h) for h in ids[eligible&(availability>=.95)] if post[str(h)]!="2259"])
 lookup={int(h):i for i,h in enumerate(ids)}
 features=np.column_stack([np.log(load),np.log(capacity)])
 scale=np.nanstd(features[eligible],axis=0);scaled=features/scale
 G=nx.DiGraph();G.add_node("sink",demand=len(geo))
 for j,h in enumerate(geo):G.add_node("slot"+str(j),demand=-1)
 for h in candidates:
  hi=lookup[int(h)];G.add_edge("h"+str(h),"p"+post[str(h)],capacity=1,weight=0)
  for j,g in enumerate(geo):
   gi=lookup[int(g)]
   if role[hi]!=role[gi]:continue
   cost=float(np.sum((scaled[hi]-scaled[gi])**2))
   G.add_edge("slot"+str(j),"h"+str(h),capacity=1,weight=int(round(cost*1e8))*1000+int(h))
 # Fixed maximum two selected households per postcode ensures dispersion.
 for p in sorted({post[str(h)] for h in candidates}):G.add_edge("p"+p,"sink",capacity=2,weight=0)
 cost,flow=nx.network_simplex(G)
 matched=np.array([int(next(k for k,v in flow["slot"+str(j)].items() if v).removeprefix("h")) for j in range(len(geo))])
 pairs=[]
 for j,(g,h) in enumerate(zip(geo,matched)):
  gi,hi=lookup[int(g)],lookup[int(h)]
  pairs.append(dict(node_index=j,geo_household=int(g),matched_household=int(h),matched_postcode=post[str(h)],
    model_role=int(role[gi]),geo_pretest_load_kwh_day=float(load[gi]),matched_pretest_load_kwh_day=float(load[hi]),
    geo_pv_kwp=float(capacity[gi]),matched_pv_kwp=float(capacity[hi]),
    standardized_log_feature_distance=float(np.linalg.norm(scaled[gi]-scaled[hi]))))
 result=dict(status="frozen_before_replay",created_utc=utcnow(),n_households=27,geo_ids=geo.tolist(),matched_ids=matched.tolist(),
   match_method="Minimum-cost flow matching on standardized log pretest gross daily demand and median nameplate PV capacity; exact known/heldout role; outside postcode2259; maximum2 per postcode; pretest feeder availability>=95%; no test outcomes or coverage used to select.",
   selected_postcode_count=len({post[str(h)] for h in matched}),pairs=pairs,
   boundary="Equal-size and partially load/PV matched constructed deployments on same first27 points; common dates. Residual matching imbalance is reported, not assumed absent. Postcodes do not establish real feeders.",
   standardized_mean_difference_load=float((np.mean(load[[lookup[int(h)] for h in matched]])-np.mean(load[[lookup[int(h)] for h in geo]]))/np.nanstd(load[eligible])),
   standardized_mean_difference_capacity=float((np.mean(capacity[[lookup[int(h)] for h in matched]])-np.mean(capacity[[lookup[int(h)] for h in geo]]))/np.nanstd(capacity[eligible])))
 save_json(OUT/"cohort_plan.json",result)
 return geo,matched,result
def run_matched(objective,ids):
 path=OUT/(objective+"_matched_dispatch.npz")
 if path.exists():return dict(np.load(path))
 date,actual,loadforecast,pvforecast,clean,finite,prov=read_inputs(objective,ids,11,
  decision_arrays="datasets/decision_arrays_physical.npz",msecont_arrays="datasets/mse_continuation_arrays_physical.npz",night_zero=True)
 gc,pv,cl=[actual[k] for k in ["truth_gc_kw","truth_pv_kw","truth_cl_kw"]]
 qp=DispatchQP(Battery(),"tou",tolerance=1e-13);buy,sell=tariff("tou")
 charge=np.full_like(gc,np.nan);discharge=np.full_like(gc,np.nan);bill=np.full(gc.shape[:2],np.nan);errors=[]
 for h in range(len(ids)):
  for d in range(len(date)):
   if not finite[h,d] or not np.isfinite(loadforecast[h,d]+pvforecast[h,d]).all():continue
   try:
    c,dc,_=qp.solve(loadforecast[h,d]-pvforecast[h,d])
    charge[h,d]=c;discharge[h,d]=dc
    bill[h,d]=realized_bill(gc[h,d]-pv[h,d],c,dc,buy,sell)+.5*.18*cl[h,d].sum()
   except Exception as e:errors.append(dict(household=int(ids[h]),date=str(date[d]),error=str(e)))
 if errors:
  save_json(OUT/(objective+"_errors.json"),errors);raise ValueError("Dispatch errors retained")
 arr=dict(date=date,household_id=ids,load_kw=gc+cl,pv_kw=pv,charge_kw=charge,discharge_kw=discharge,
          bill_byhouse_aud=bill.T,clean_day=clean.all(axis=0))
 np.savez_compressed(path,**arr);return arr
def replay_matched(objective,x,mask):
 D=len(x["date"]);H=27
 def pad(ar):
  full=np.zeros((D*48,55));full[:,:H]=ar.transpose(1,2,0).reshape(D*48,H);return full
 load,pv,battery=pad(x["load_kw"]),pad(x["pv_kw"]),pad(x["charge_kw"]-x["discharge_kw"])
 model=EuropeanLV();model.dss.Text.Command('set datapath="'+str(OUT)+'"')
 result=model.run(load,pv,battery,return_bus=True);v=result["household_voltage_pu"][:,:H]
 arr={k:value for k,value in result.items() if isinstance(value,np.ndarray)}
 arr.update(household_voltage_pu=v,date=x["date"],date_slot=np.repeat(x["date"],48),household_id=x["household_id"],
  voltage_household_id=x["household_id"],battery_installed=np.ones(H,bool),clean_day=mask,
  original_cohort_clean_day=x["clean_day"],physical_input_complete_day=result["input_valid"].reshape(D,48).all(axis=1),
  converged_day=result["converged"].reshape(D,48).all(axis=1),bill_byhouse_aud=x["bill_byhouse_aud"],
  bill_mean_aud=x["bill_byhouse_aud"].mean(axis=1),
  power_balance_residual_kw=result["source_import_kw"]-(load-pv+battery).sum(axis=1)-result["loss_kw"])
 arr["voltage_violation_fraction"]=np.where(result["converged"],((v<.95)|(v>1.05)).mean(axis=1),np.nan)
 arr["voltage_exceedance_pu"]=(np.maximum(.95-v,0)+np.maximum(v-1.05,0)).sum(axis=1)
 arr["voltage_violation_fraction_090_110"]=np.where(result["converged"],((v<.90)|(v>1.10)).mean(axis=1),np.nan)
 return arr
def main():
 OUT.mkdir(parents=True,exist_ok=True);REPLAY.mkdir(parents=True,exist_ok=True)
 geo,matched,plan=choose()
 old={};new={}
 for objective in ["msecont","dfl","mlp"]:
  p=ROOT/"results/geographic_replay/geographic_physical"/f"{objective}_seed11_postcode2259_protocol_eligible_b27_all_tou_test_physical.npz"
  old[objective]=dict(np.load(p));new[objective]=run_matched(objective,matched)
 mask=old["msecont"]["clean_day"]&new["msecont"]["clean_day"]
 for objective in old:
  assert np.array_equal(old[objective]["household_id"],geo)
  assert np.array_equal(new[objective]["date"],old[objective]["date"])
  assert np.array_equal(new[objective]["clean_day"],new["msecont"]["clean_day"])
  for cohort,arr in [("geo_narrow",old[objective]),("geo_matched",replay_matched(objective,new[objective],mask))]:
   sid=f"{objective}_seed11_{cohort}_b27_all_tou_eulv_matched_dates_revision"
   path=REPLAY/(sid+".npz")
   if cohort=="geo_narrow":
    arr["original_cohort_clean_day"]=arr["clean_day"].copy();arr["clean_day"]=mask
    arr["battery_installed"]=np.ones(27,bool)
   np.savez_compressed(path,**arr)
   meta=dict(status="completed",scenario=sid,objective=objective,seed=11,panel=cohort,battery_count=27,tariff_mix="all_tou",
     parameters=dict(network="eulv",source_pu=1.05,power_factor=.95,mapping_seed=None),output=str(path),
     n_clean_dates=int(mask.sum()),n_households=27,cohort_plan_sha256=sha256(OUT/"cohort_plan.json"),output_sha256=sha256(path),
     source="Original completed geo27 replay with newly intersected dates" if cohort=="geo_narrow" else "New pretest-matched geographically dispersed27 replay",
     solver_failed_slots=int((arr["input_valid"]&~arr["converged"]).sum()),
     max_power_balance_residual_kw=float(np.nanmax(np.abs(arr["power_balance_residual_kw"]))))
   save_json(path.with_suffix(".json"),meta);print(json.dumps({"scenario":sid,"clean_days":int(mask.sum())}),flush=True)
 save_json(OUT/"status.json",dict(status="completed",n_common_clean_days=int(mask.sum()),
  common_clean_dates=new["msecont"]["date"][mask].tolist(),plan=plan))
if __name__=="__main__":main()
