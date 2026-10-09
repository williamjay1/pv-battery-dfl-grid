"""Revision voltage distributions and engineering-scale diagnostics (no fitting)."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
from network_replay import sha256
from revision_network_io import save_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"revision_20261009/network_diagnostics"
Q=np.array([0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1])
UP=np.round(np.arange(.98,1.10001,.0025),7)
LOW=np.round(np.arange(.90,1.02001,.0025),7)
def f(x):
 return float(x) if np.isfinite(x) else None
def ratio(a,b): return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)>0)
def longest(mask):
 x=np.r_[False,mask,False].astype(int);change=np.diff(x)
 lengths=np.flatnonzero(change==-1)-np.flatnonzero(change==1)
 return float(lengths.max(initial=0)*.5),int(len(lengths)),float(np.quantile(lengths,.95)*.5) if len(lengths) else 0.
def quant(x):return np.quantile(x,Q).tolist() if len(x) else [None]*len(Q)
def one(path):
 with np.load(path) as z: a={k:z[k] for k in z.files}
 meta=json.loads(path.with_suffix(".json").read_text("utf-8"))
 sid=path.stem
 clean=a["clean_day"]&a["converged_day"]
 D=len(a["date"]);v=a["household_voltage_pu"].reshape(D,48,-1).astype(float);H=v.shape[2]
 if not clean.any(): raise ValueError("No clean dates")
 valid=np.repeat(clean,48);vv=v[clean].reshape(-1,H);obs=vv.ravel()
 over=v>1.05;under=v<.95
 oe=np.maximum(v-1.05,0);ue=np.maximum(.95-v,0)
 over_n=over.sum(axis=(1,2));under_n=under.sum(axis=(1,2))
 days={}
 days["daily_over_pct"]=over.mean(axis=(1,2))*100
 days["daily_under_pct"]=under.mean(axis=(1,2))*100
 days["daily_voltage_pct"]=days["daily_over_pct"]+days["daily_under_pct"]
 days["daily_over_110_pct"]=(v>1.10).mean(axis=(1,2))*100
 days["daily_under_090_pct"]=(v<.90).mean(axis=(1,2))*100
 days["daily_voltage_090_110_pct"]=days["daily_over_110_pct"]+days["daily_under_090_pct"]
 days["daily_excursion_pu"]=(oe+ue).mean(axis=(1,2))
 days["daily_conditional_over_pu"]=ratio(oe.sum(axis=(1,2)),over_n)
 days["daily_conditional_under_pu"]=ratio(ue.sum(axis=(1,2)),under_n)
 source=a["source_import_kw"].reshape(D,48)
 days["daily_import_peak_kw"]=np.maximum(source,0).max(axis=1)
 days["daily_export_peak_kw"]=np.maximum(-source,0).max(axis=1)
 days["daily_loss_kwh"]=a["loss_kw"].reshape(D,48).sum(axis=1)*.5
 days["daily_transformer_peak_loading_pct"]=a["transformer_loading_pu"].reshape(D,48).max(axis=1)*100
 if "transformer_max_phase_loading_pu" in a:
  days["daily_transformer_max_phase_loading_pct"]=a["transformer_max_phase_loading_pu"].reshape(D,48).max(axis=1)*100
 days["daily_operating_cost_aud"]=a["bill_byhouse_aud"].mean(axis=1)
 for k in days: days[k]=np.where(clean,days[k],np.nan)
 nodeover=(vv>1.05).mean(axis=0)*100;nodeunder=(vv<.95).mean(axis=0)*100
 allflat=v.reshape(-1,H)
 events={}
 for label,mask in [("over",allflat>1.05),("under",allflat<.95)]:
  x=np.array([longest(mask[:,h]&valid) for h in range(H)])
  events["node_longest_"+label+"_hours"]=x[:,0]
  events["node_"+label+"_episode_count"]=x[:,1]
  events["node_"+label+"_episode_p95_hours"]=x[:,2]
 ocount=int((vv>1.05).sum());ucount=int((vv<.95).sum())
 widths=np.array([.0001,.0005,.001,.0025,.005])
 near={str(t):[float(np.mean(np.abs(obs-t)<=w)*100) for w in widths] for t in [.90,.95,1.05,1.10]}
 voltage=dict(quantile_levels=Q.tolist(),quantiles=quant(obs),over_pct=float(100*ocount/obs.size),
  under_pct=float(100*ucount/obs.size),conditional_over_pu=f(np.maximum(obs-1.05,0).sum()/ocount) if ocount else None,
  over_110_pct=float(np.mean(obs>1.10)*100),under_090_pct=float(np.mean(obs<.90)*100),
  conditional_under_pu=f(np.maximum(.95-obs,0).sum()/ucount) if ucount else None,
  mean_excursion_pu=float((np.maximum(.95-obs,0)+np.maximum(obs-1.05,0)).mean()),
  worst_over_node_index=int(np.argmax(nodeover)),worst_under_node_index=int(np.argmax(nodeunder)),
  lowest_voltage_node_index=int(np.argmin(vv.min(axis=0))),highest_voltage_node_index=int(np.argmax(vv.max(axis=0))),
  near_threshold_halfwidth_pu=widths.tolist(),near_threshold_mass_pct=near,
  precision="Stored historical voltages float32; threshold ambiguity <=6e-8 pu. Curves use declared fixed thresholds; original float64 scalar exposure also retained.")
 months=np.array([str(d)[:7] for d in a["date"]])
 coverage={m:dict(calendar_days=int((months==m).sum()),clean_days=int((clean&(months==m)).sum())) for m in np.unique(months)}
 arrays=dict(date=a["date"],clean_day=clean,household_id=a["household_id"],
   voltage_household_id=a.get("voltage_household_id",a["household_id"]),
   voltage_quantile_levels=Q,voltage_quantiles=np.quantile(obs,Q),
   threshold_upper_pu=UP,threshold_lower_pu=LOW,
   upper_exposure_pct=np.array([(obs>u).mean()*100 for u in UP]),
   lower_exposure_pct=np.array([(obs<l).mean()*100 for l in LOW]),
   hour_over_pct=(v[clean]>1.05).mean(axis=(0,2))*100,
   hour_under_pct=(v[clean]<.95).mean(axis=(0,2))*100,
   hour_voltage_mean_pu=v[clean].mean(axis=(0,2)),
   node_over_pct=nodeover,node_under_pct=nodeunder,node_voltage_quantiles=np.quantile(vv,Q,axis=0).T,
   node_mean_voltage_pu=vv.mean(axis=0),bill_byhouse_aud=a["bill_byhouse_aud"],
   battery_installed=a.get("battery_installed",np.ones(H,bool)),**events,**days)
 arrays["daily_original_voltage_pct"]=np.where(clean,a["voltage_violation_fraction"].reshape(D,48).mean(axis=1)*100,np.nan)
 outpath=OUT/(sid+".npz");np.savez_compressed(outpath,**arrays)
 metrics={k:dict(mean=f(np.nanmean(x)) if np.isfinite(x).any() else None,quantiles=quant(x[np.isfinite(x)]),
   aggregation="Unweighted mean of day-specific conditional severity over days with events; not pooled severity" if "conditional" in k else "Mean across common clean dates") for k,x in days.items()}
 return dict(diagnostic_version=2,scenario_id=sid,source=str(path.relative_to(ROOT)),source_sha256=sha256(path),
   metadata={k:meta.get(k) for k in ["objective","base_objective","comparison_group","seed","panel","battery_count","tariff_mix","parameters","network_metadata"]},
   coverage=dict(clean_days=int(clean.sum()),physical_complete_days=int(a["converged_day"].sum()),months=coverage,
                 n_customers=H,n_clean_customer_halfhours=int(obs.size)),voltage=voltage,flow=metrics,
   arrays=str(outpath.relative_to(ROOT)))
BOOT_CACHE={}
def bootstrap(delta,reps=2000):
 T=len(delta);key=(T,reps)
 if key not in BOOT_CACHE:
  starts=np.random.default_rng(20261005).integers(0,T,size=(reps,int(np.ceil(T/7))))
  BOOT_CACHE[key]=((starts[:,:,None]+np.arange(7))%T).reshape(reps,-1)[:,:T]
 samples=delta[BOOT_CACHE[key]]
 out=np.nanmean(samples,axis=1);out=out[np.isfinite(out)]
 return dict(mean=f(np.nanmean(delta)),ci95=np.quantile(out,[.025,.975]).tolist(),n_clean_days=int(np.isfinite(delta).sum()))
def comparisons(records):
 groups={}
 for r in records:
  m=r["metadata"];p=m.get("parameters") or {}
  if not m.get("objective") or m["objective"]=="none": continue
  key=(m.get("panel"),m.get("battery_count"),m.get("tariff_mix"),p.get("network"),p.get("source_pu"),
       p.get("mapping_seed"),p.get("power_factor",.95),p.get("impedance_scale",1),p.get("export_limit_kw"),m.get("comparison_group") or "original_physical")
  groups.setdefault(key,{}).setdefault(m.get("base_objective") or m["objective"],{})[m.get("seed",11)]=r
 rows=[]
 for key,methods in groups.items():
  if "dfl" not in methods:continue
  for b in sorted(m for m in methods if m!="dfl"):
   if b not in methods:continue
   seeds=sorted(set(methods["dfl"])&set(methods[b]))
   if not seeds:continue
   aa=[];bb=[]
   for seed in seeds:
    aa.append(dict(np.load(ROOT/methods["dfl"][seed]["arrays"])))
    bb.append(dict(np.load(ROOT/methods[b][seed]["arrays"])))
   for a,z in zip(aa,bb):
    assert np.array_equal(a["date"],z["date"]) and np.array_equal(a["clean_day"],z["clean_day"])
   outcomes={}
   for metric in [k for k in aa[0] if k.startswith("daily_") and "conditional" not in k]:
    bv=np.mean([x[metric] for x in bb],axis=0);dv=np.mean([x[metric] for x in aa],axis=0)
    baseline=float(np.nanmean(bv));r=bootstrap(dv-bv)
    outcomes[metric]=dict(baseline_mean=baseline,dfl_mean=f(np.nanmean(dv)),delta=r,
       relative_delta_pct=f(r["mean"]/baseline*100) if baseline else None)
   mask=aa[0]["clean_day"];H=len(aa[0]["household_id"])
   cost=np.mean([x["bill_byhouse_aud"] for x in aa],axis=0)-np.mean([x["bill_byhouse_aud"] for x in bb],axis=0)
   dh=cost[mask].mean(axis=0)
   vdelta=np.mean([x["node_over_pct"]+x["node_under_pct"] for x in aa],axis=0)-np.mean([x["node_over_pct"]+x["node_under_pct"] for x in bb],axis=0)
   idv=aa[0]["voltage_household_id"];idh=aa[0]["household_id"];installed=aa[0]["battery_installed"]
   owner=np.isin(idv,idh[installed])
   distributions=dict(household_id=idh.tolist(),cost_delta_aud_per_day=dh.tolist(),cost_delta_quantiles=quant(dh),
      cost_benefiting_fraction=float(np.mean(dh< -1e-8)),cost_harmed_fraction=float(np.mean(dh>1e-8)),
      cost_owner_benefiting_fraction=float(np.mean(dh[installed]< -1e-8)) if installed.any() else None,
      cost_owner_harmed_fraction=float(np.mean(dh[installed]>1e-8)) if installed.any() else None,
      cost_owner_delta_quantiles=quant(dh[installed]),
      voltage_household_id=idv.tolist(),voltage_delta_pp=vdelta.tolist(),
      owner_voltage_delta_mean_pp=f(vdelta[owner].mean()) if owner.any() else None,
      nonowner_voltage_delta_mean_pp=f(vdelta[~owner].mean()) if (~owner).any() else None,
      voltage_benefiting_fraction=float((vdelta< -1e-7).mean()),voltage_harmed_fraction=float((vdelta>1e-7).mean()))
   rows.append(dict(key=list(key[:9]),comparison_group=key[9],reference=b,target="dfl",seeds=seeds,n_customers=H,
       outcomes=outcomes,households=distributions,interpretation="Paired 7-day calendar blocks; fixed constructed cohort; no population or causal inference. Conditional-severity ratios remain descriptive."))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument("--scope",choices=["all","reference"],default="all");a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 files=[]
 for folder in ["main_physical","heldout_physical","none_physical","external_full_physical","mapping_full_physical","strong_baselines_physical"]:
  for path in sorted((ROOT/"results/network_replay"/folder).glob("*.npz")):
   if a.scope=="reference" and "seed11_panel0_b55_all_tou" not in path.name:continue
   files.append(path)
 files+=sorted((ROOT/"revision_20261009/network_replay").glob("*/*.npz"))
 files=[p for p in files if p.with_suffix(".json").exists() and json.loads(p.with_suffix(".json").read_text()).get("status")=="completed"]
 previous=json.loads((OUT/"scenarios.json").read_text())["records"] if (OUT/"scenarios.json").exists() else []
 cache={r["source"]:r for r in previous};records=[]
 for i,path in enumerate(files):
  old=cache.get(str(path.relative_to(ROOT)))
  if old and old.get("diagnostic_version")==2 and old["source_sha256"]==sha256(path) and (ROOT/old["arrays"]).exists():
   records.append(old)
  else: records.append(one(path))
  if i%12==0:print(json.dumps({"completed":i+1,"total":len(files),"scenario":path.stem}),flush=True)
 save_json(OUT/"scenarios.json",dict(schema_version=1,scope=a.scope,quantile_levels=Q.tolist(),records=records))
 save_json(OUT/"paired.json",dict(schema_version=1,primary_reference="msecont",records=comparisons(records)))
 print(json.dumps({"status":"completed","scenarios":len(records)}),flush=True)
if __name__=="__main__":main()
