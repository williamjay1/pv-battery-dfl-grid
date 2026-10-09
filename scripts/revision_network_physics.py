"""Post-review physical brackets and uniform PCC export-limiter replay."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,re,time,traceback
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ.setdefault(k,"1")
import numpy as np
from network_model import EuropeanLV
from network_replay import sha256,utcnow
from revision_network_io import save_json
from battery_model import tariff
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"revision_20261009";OUT=BASE/"network_replay"
def prepare_model(scale):
 if scale==1:return ROOT/"datasets/network_eulv"
 out=BASE/("network_models/z"+str(scale).replace(".","p"));out.mkdir(parents=True,exist_ok=True)
 for name in ["LineCode.txt","Loads.txt","Transformers.txt","Lines.txt"]:
  source=(ROOT/"datasets/network_eulv"/name).read_text()
  if name=="Lines.txt":
   source=re.sub(r"(?i)(length=)([0-9.eE+-]+)",lambda m:m[1]+format(float(m[2])*scale,".12g"),source)
  path=out/name
  if path.exists() and path.read_text()!=source:raise ValueError("Existing model copy differs")
  if not path.exists():path.write_text(source)
 return out
def inputs(objective,seed=11,panel=0,dispatch_path=None):
 splits=json.loads((ROOT/"datasets/splits_v1.json").read_text())
 ids=splits["primary_panels"][panel] if panel<3 else splits["heldout_network_household_ids"]
 path=Path(dispatch_path) if dispatch_path else ROOT/"results/dispatch"/f"{objective}_tou_seed{seed}_test_physical.npz"
 report=json.loads(path.with_suffix(".json").read_text())
 if report.get("failed",report.get("n_failed",0)) or report.get("failures") or report.get("status")=="failed":raise ValueError("Dispatch failed")
 with np.load(path) as a:
  h=[int(np.flatnonzero(a["household_id"]==x)[0]) for x in ids]
  vals={k:a[k][h].astype(float) for k in ["charge_kw","discharge_kw","actual_load_kw","actual_pv_kw","bill_aud"]}
  vals["clean_day"]=a["clean_day"][h].all(axis=0);vals["date"]=a["date"].copy()
 vals["household_id"]=np.array(ids);vals["path"]=path
 finite=np.isfinite(vals["actual_load_kw"]+vals["actual_pv_kw"]+vals["charge_kw"]+vals["discharge_kw"]).all(axis=(0,2))
 if np.any(vals["clean_day"]&~finite):raise ValueError("Source-clean dates contain missing injections/actions")
 return vals
def run_case(config):
 start=time.perf_counter();objective=config["objective"]
 panel=config.get("panel",0)
 x=inputs(objective,panel=panel,dispatch_path=config.get("dispatch_path"));D=len(x["date"]);H=len(x["household_id"])
 scale=config.get("impedance_scale",1.);pf=config.get("power_factor",.95);source=config.get("source_pu",1.05)
 limit=config.get("export_limit_kw")
 folder=OUT/config["suite"];folder.mkdir(parents=True,exist_ok=True)
 suffix=f"_z{scale}_pf{pf}_cap{limit}".replace(".","p")
 adoption=0 if objective=="none" else 55
 sid=f"{objective}_seed11_panel{panel}_b{adoption}_all_tou_eulv_src{source}_fixed_test_revision"+suffix
 path=folder/(sid+".npz");statuspath=path.with_suffix(".json")
 if path.exists() and statuspath.exists() and json.loads(statuspath.read_text()).get("status")=="completed":
  cached=json.loads(statuspath.read_text())
  if cached['input_sha256']!=sha256(x['path']):raise ValueError('Completed replay input changed; use explicit new-version output')
  return cached
 params=dict(network="eulv",source_pu=source,power_factor=pf,impedance_scale=scale,export_limit_kw=limit,mapping_seed=None)
 st=dict(status="running",scenario=sid,objective=objective,seed=11,panel=panel,battery_count=adoption,tariff_mix="all_tou",
  parameters=params,input=str(x["path"]),input_sha256=sha256(x["path"]),started_utc=utcnow(),output=str(path),
  comparison_group=config.get("comparison_group","original_physical"),base_objective=config.get("base_objective",objective),
  script_sha256=sha256(__file__))
 save_json(statuspath,st)
 try:
  shape=lambda ar:ar.transpose(1,2,0).reshape(D*48,H)
  load=shape(x["actual_load_kw"]);pv=shape(x["actual_pv_kw"])
  battery=shape(x["charge_kw"]-x["discharge_kw"])
  curtailed=np.zeros_like(pv)
  if limit is not None:
   curtailed=np.minimum(pv,np.maximum(pv-load-battery-limit,0))
   pv=pv-curtailed
   finite=np.isfinite(load+pv+battery).all(axis=1)
   if np.any((-load[finite]+pv[finite]-battery[finite])>limit+1e-8):raise ValueError("Export cap not enforced")
  model=EuropeanLV(model_dir=prepare_model(scale),source_pu=source)
  model.dss.Text.Command('set datapath="'+str(BASE/"network_scratch")+'"')
  out=model.run(load,pv,battery,power_factor=pf,return_bus=True)
  arrays={k:v for k,v in out.items() if isinstance(v,np.ndarray)}
  # Curtailment happens only while the aggregate PCC exports. Thus the general
  # meter still exports after clipping and lost retail revenue is sell*energy.
  lost_export_revenue=.5*.08*curtailed.reshape(D,48,H).sum(axis=1)
  bills=x["bill_aud"].T+lost_export_revenue
  arrays.update(date=x["date"],date_slot=np.repeat(x["date"],48),household_id=x["household_id"],
   voltage_household_id=x["household_id"],battery_installed=np.full(H,adoption>0,bool),tariff_is_tou=np.ones(H,bool),
   clean_day=x["clean_day"],physical_input_complete_day=out["input_valid"].reshape(D,48).all(axis=1),
   converged_day=out["converged"].reshape(D,48).all(axis=1),bill_byhouse_aud=bills,bill_mean_aud=bills.mean(axis=1),
   power_balance_residual_kw=out["source_import_kw"]-(load-pv+battery).sum(axis=1)-out["loss_kw"],
   curtailment_kw=curtailed,curtailment_kwh_byhouse=.5*curtailed.reshape(D,48,H).sum(axis=1),
   lost_export_revenue_aud_byhouse=lost_export_revenue)
  v=out["household_voltage_pu"]
  arrays["voltage_violation_fraction_090_110"]=np.where(out["converged"],((v<.90)|(v>1.10)).mean(axis=1),np.nan)
  np.savez_compressed(path,**arrays)
  clean=arrays["clean_day"]
  st.update(status="completed",finished_utc=utcnow(),elapsed_seconds=time.perf_counter()-start,
    network_metadata=out["metadata"],n_clean_dates=int(clean.sum()),output_sha256=sha256(path),
    solver_failed_slots=int((out["input_valid"]&~out["converged"]).sum()),
    max_balance_residual_kw=float(np.nanmax(np.abs(arrays["power_balance_residual_kw"]))),
    mean_daily_curtailment_kwh=float(arrays["curtailment_kwh_byhouse"][clean].sum(axis=1).mean()),
    mean_lost_export_revenue_aud_per_house_day=float(lost_export_revenue[clean].mean()),
    control_interpretation="Fixed precomputed battery actions and SOC; ideal PCC active-power export limiter curtails available PV only; PV/battery unity PF; general meter settlement recalculated after curtailment. No claim of full inverter standard compliance.",
    uncertainty_interpretation="Declared deterministic single-factor engineering brackets, not confidence intervals on measured network parameters. Scaling cable length scales cable R/X (zero capacitance), not transformer impedance.")
  save_json(statuspath,st);print(json.dumps(dict(scenario=sid,status=st["status"],elapsed=st["elapsed_seconds"])),flush=True)
  return st
 except Exception as e:
  st.update(status="failed",error=str(e),traceback=traceback.format_exc());save_json(statuspath,st);raise
def main():
 p=argparse.ArgumentParser();p.add_argument("--suite",choices=["physical_brackets","export_limit","new_dispatch"],required=True)
 p.add_argument("--workers",type=int,default=3);p.add_argument("--dispatch-path");p.add_argument("--objective")
 p.add_argument("--panels",type=int,nargs="+",default=[0])
 a=p.parse_args()
 if a.suite=="physical_brackets":
  tasks=[dict(suite=a.suite,objective=m,**q) for m in ["msecont","dfl","mlp"] for q in [
    dict(power_factor=.90),dict(power_factor=1.),dict(impedance_scale=.90),dict(impedance_scale=1.10)]]
 elif a.suite=="export_limit":
  tasks=[dict(suite=a.suite,objective=m,export_limit_kw=3.,panel=panel) for panel in a.panels for m in ["msecont","dfl","mlp","hgb","calibrated","stochastic","self","none","oracle"]]
 else:tasks=[dict(suite=a.suite,objective=a.objective,dispatch_path=a.dispatch_path,panel=panel) for panel in a.panels]
 for t in tasks:prepare_model(t.get("impedance_scale",1.))
 (BASE/"network_scratch").mkdir(parents=True,exist_ok=True)
 status=dict(status="running",tasks=tasks,results=[],started_utc=utcnow())
 dest=BASE/("network_"+a.suite+"_status.json");save_json(dest,status)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(run_case,t):t for t in tasks}
  for fut in concurrent.futures.as_completed(fs):
   try:status["results"].append(fut.result())
   except Exception as e:status["results"].append(dict(status="failed",task=fs[fut],error=str(e)))
   save_json(dest,status)
 status["status"]="completed" if all(x["status"]=="completed" for x in status["results"]) else "failed"
 status["finished_utc"]=utcnow();save_json(dest,status)
 print(json.dumps({"status":status["status"],"results":len(status["results"])}),flush=True)
if __name__=="__main__":main()
