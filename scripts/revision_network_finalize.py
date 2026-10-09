"""Summarize and independently check revision-only network artifacts."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from network_replay import sha256,utcnow
from revision_network_io import save_json
from revision_network_diagnostics import comparisons,bootstrap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"revision_20261009"
def main():
 source=OUT/"network_diagnostics/scenarios.json";report=json.loads(source.read_text())
 for r in report["records"]:
  r["voltage"]["conditional_aggregation"]="Pooled sum of exceedance magnitudes / pooled number of corresponding events, across common clean customer half-hours. Undefined if no event."
  for k,v in r["flow"].items():
   v["aggregation"]="Unweighted mean of daily conditional severity over event-bearing days; not pooled severity" if "conditional" in k else "Mean across common clean dates"
 save_json(source,report)
 fixed=comparisons([r for r in report["records"] if r["metadata"]["seed"]==11])
 save_json(OUT/"network_diagnostics/fixed_seed11_paired.json",dict(schema_version=1,seeds=[11],records=fixed,
    purpose="Uniform seed11 comparison across operating conditions; main paired.json still averages all three seeds where present."))
 precision=[]
 for r in report['records']:
  for side in ['over','under']:
   severity=r['voltage']['conditional_'+side+'_pu']
   if r['voltage'][side+'_pct']==0:assert severity is None,r['scenario_id']
   else:assert severity is not None and severity>0,r['scenario_id']
  with np.load(ROOT/r['arrays']) as a:
   difference=float(np.nanmean(a['daily_voltage_pct'])-np.nanmean(a['daily_original_voltage_pct']))
  precision.append(dict(scenario=r['scenario_id'],difference_pp=difference))
 precision_report=dict(max_absolute_difference_of_clean_mean_pp=max(abs(r['difference_pp']) for r in precision),
  nonzero_records=[r for r in precision if abs(r['difference_pp'])>1e-8],
  interpretation='Reconstructed historical float32 node threshold fraction minus original stored float64 count fraction. Rounding near fixed band edge, not new physical runs.')
 save_json(OUT/'network_threshold_precision_audit.json',precision_report)
 checks=[];pairs={}
 for p in sorted((OUT/"network_replay").glob("*/*.npz")):
  j=json.loads(p.with_suffix(".json").read_text())
  if j.get("status")!="completed":continue
  with np.load(p) as a:
   clean=a["clean_day"];D=len(clean);H=len(a["household_id"]);v=a["household_voltage_pu"]
   assert v.shape==(D*48,H),(p.name,v.shape)
   assert np.all(a["converged_day"][clean]),p.name
   assert np.isfinite(v.reshape(D,48,H)[clean]).all(),p.name
   assert np.isfinite(a["bill_byhouse_aud"][clean]).all(),p.name
   failure=int((a["input_valid"]&~a["converged"]).sum())
   residual=float(np.nanmax(np.abs(a["power_balance_residual_kw"])))
   assert failure==0 and residual<1e-3,(p.name,failure,residual)
   if j["parameters"].get("export_limit_kw") is not None:
    original=dict(np.load(j["input"]));index=[int(np.flatnonzero(original["household_id"]==h)[0]) for h in a["household_id"]]
    flat=lambda x:x[index].transpose(1,2,0).reshape(D*48,H).astype(float)
    gross=flat(original["actual_load_kw"]);pv=flat(original["actual_pv_kw"])
    bat=flat(original["charge_kw"])-flat(original["discharge_kw"]);c=a["curtailment_kw"]
    valid=a["input_valid"]
    assert np.all(c[valid]>=-1e-10) and np.all(c[valid]<=pv[valid]+1e-10)
    assert np.max(pv[valid]-c[valid]-gross[valid]-bat[valid])<=3+1e-8
    exp=original["bill_aud"][index].T+.5*.08*c.reshape(D,48,H).sum(axis=1)
    assert np.allclose(exp[clean],a["bill_byhouse_aud"][clean],rtol=0,atol=1e-10)
   checks.append(dict(file=str(p.relative_to(ROOT)),H=H,clean_days=int(clean.sum()),solver_failed_slots=failure,max_balance_kw=residual,sha256=sha256(p),
    reused_original_geo_ac=j.get('source')=='Original completed geo27 replay with newly intersected dates'))
 group={}
 for r in report["records"]:
  m=r["metadata"]
  if m["panel"] in ["geo_narrow","geo_matched"] and m["objective"] in ["msecont","dfl"]:
   group[(m["panel"],m["objective"])]=dict(np.load(ROOT/r["arrays"]))
 geo=[]
 if len(group)==4:
  for metric in ["daily_operating_cost_aud","daily_over_pct","daily_under_pct","daily_import_peak_kw","daily_export_peak_kw","daily_loss_kwh"]:
   narrow=group[("geo_narrow","dfl")][metric]-group[("geo_narrow","msecont")][metric]
   broad=group[("geo_matched","dfl")][metric]-group[("geo_matched","msecont")][metric]
   assert np.array_equal(np.isfinite(narrow),np.isfinite(broad))
   geo.append(dict(metric=metric,narrow_method_gap=float(np.nanmean(narrow)),matched_dispersed_method_gap=float(np.nanmean(broad)),difference_of_method_gaps=bootstrap(narrow-broad)))
 save_json(OUT/"network_geography/paired_cohort_contrast.json",dict(contrast="(DFL-continuedMSE) in narrow27 minus same gap in matched dispersed27",records=geo,
    interpretation="Descriptive difference of method gaps under matched sizes, nodes and dates. Not causal effect of geographic clustering or known real feeder connectivity."))
 result=dict(status="PASS",created_utc=utcnow(),new_scenarios=len(checks),records=checks,
   newly_executed_ac_replays=sum(not r['reused_original_geo_ac'] for r in checks),
   reused_original_geo_replays_with_new_common_mask=sum(r['reused_original_geo_ac'] for r in checks),
   pass_definition="Numerical convergence, finite source-clean outputs, power balance and export-settlement consistency. PASS does not mean network constraints are never violated or establish real-feeder validity.",
   maximum_balance_residual_kw=max(x["max_balance_kw"] for x in checks),
   matching_plan="network_geography/cohort_plan.json",mechanism="network_mechanism/summary.json",
   sources="research/engineering_sources.json",
   stored_historical_voltage_precision="Float32 historical node voltages: <=6e-8pu rounding near1pu; fixed curves and direction splits are diagnostic. Original float64 counts retained in daily_original_voltage_pct.",
   coverage="Every metric uses common source-clean dates. Missing months retained as zero coverage, not imputed.",
   control_economics="Export cap curtails PV only; battery actions unchanged, preserved SOC; lost export retail revenue included separately from unchanged wear/CL.",
   files=dict(scenarios="network_diagnostics/scenarios.json",pooled_three_seed_comparisons="network_diagnostics/paired.json",uniform_seed11_comparisons="network_diagnostics/fixed_seed11_paired.json",scale="network_system_scale.json"))
 save_json(OUT/"network_revision_audit.json",result)
 print(json.dumps({"status":"PASS","new_scenarios":len(checks),"fixed_comparisons":len(fixed)}),flush=True)
if __name__=="__main__":main()
