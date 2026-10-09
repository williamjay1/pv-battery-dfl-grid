"""Fixed-slot AC sensitivity and continuous-voltage transmission diagnostics."""
from __future__ import annotations
import json,os,time
from pathlib import Path
import numpy as np
from network_model import EuropeanLV
from network_replay import save_json,sha256,utcnow
from revision_network_physics import inputs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"revision_20261009/network_mechanism"
def vh(net,p,q):
 value=np.abs(net._solve(p,q)[net.house_nodes])/net.lv_base_v
 if not net.dss.Solution.Converged():raise ValueError("AC failed")
 return value
def sensitivity(net,p,q,h):
 sp=np.empty((55,55));sq=np.empty((55,55))
 for j in range(55):
  shift=np.zeros(55);shift[j]=h
  sp[:,j]=(vh(net,p+shift,q)-vh(net,p-shift,q))/(2*h)
  sq[:,j]=(vh(net,p,q+shift)-vh(net,p,q-shift))/(2*h)
 return sp,sq
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 a=inputs("msecont");b=inputs("dfl")
 assert np.array_equal(a["date"],b["date"]) and np.array_equal(a["clean_day"],b["clean_day"])
 clean=a["clean_day"];D=len(clean);dates=a["date"]
 flat=lambda x:x.transpose(1,2,0).reshape(D*48,55)
 load=flat(a["actual_load_kw"]);pv=flat(a["actual_pv_kw"])
 ba=flat(a["charge_kw"]-a["discharge_kw"]);bb=flat(b["charge_kw"]-b["discharge_kw"])
 # Frozen independently of the DFL effect: first available clean day per
 # meteorological season, four clock slots; plus exogenous net-load quantiles.
 seasons=np.array([["summer","summer","autumn","autumn","autumn","winter","winter","winter","spring","spring","spring","summer"][int(str(d)[5:7])-1] for d in dates])
 selected=[]
 for season in ["winter","spring","summer","autumn"]:
  day=int(np.flatnonzero(clean&(seasons==season))[0])
  for slot in [4,24,36,44]:selected.append((day*48+slot,f"{season}_first_clean_{slot/2:02.0f}h"))
 exog=(load-pv).sum(axis=1);valid=np.repeat(clean,48)
 for quantile in [.05,.5,.95]:
  target=np.quantile(exog[valid],quantile);indices=np.flatnonzero(valid)
  t=int(indices[np.argmin(np.abs(exog[indices]-target))]);selected.append((t,f"exogenous_net_q{quantile}"))
 plan=dict(created_utc=utcnow(),selection="First common clean date within each season at 02:00,12:00,18:00,22:00; nearest observed p05,p50,p95 exogenous gross-load-minus-PV sample. No ranking on method effect.",
    cases=[dict(date=str(dates[t//48]),slot=t%48,label=label) for t,label in selected],
    finite_difference_kw_kvar=[.1,.05],reference="msecont",target="dfl",
    sign_convention="P is net demand-positive kW; Q is gross-demand-positive kvar. Battery/PV unity PF implies deltaQ=0.",
    sensitivity_role="Continuous voltage local Jacobian, not derivative of an exceedance indicator; full AC target verifies finite intervention error.")
 save_json(OUT/"plan.json",plan)
 net=EuropeanLV();net.dss.Text.Command('set datapath="'+str(OUT)+'"')
 q=load*np.tan(np.arccos(.95));rows=[];matrices=[];qmat=[];vbase=[];vtarget=[];linear=[];deltaP=[]
 for t,label in selected:
  p=load[t]-pv[t]+ba[t];dp=bb[t]-ba[t]
  v0=vh(net,p,q[t]);v1=vh(net,p+dp,q[t])
  sp,sq=sensitivity(net,p,q[t],.1);sp2,sq2=sensitivity(net,p,q[t],.05)
  approx=sp@dp;actual=v1-v0;error=actual-approx
  contrib=sp*dp[None,:];phases=net.house_phases
  phase_signed=np.array([[contrib[np.ix_(phases==r,phases==c)].sum() for c in [1,2,3]] for r in [1,2,3]])
  # Mean voltage contribution per receiving node, with all sender columns.
  phase_signed=np.array([phase_signed[i]/np.sum(phases==r) for i,r in enumerate([1,2,3])])
  rows.append(dict(label=label,date=str(dates[t//48]),slot=int(t%48),
    exogenous_net_kw=float(exog[t]),battery_delta_aggregate_kw=float(dp.sum()),
    actual_voltage_delta_mean_pu=float(actual.mean()),actual_voltage_delta_max_abs_pu=float(np.abs(actual).max()),
    approximation_rmse_pu=float(np.sqrt(np.mean(error**2))),approximation_max_abs_error_pu=float(np.abs(error).max()),
    relative_l2_error=float(np.linalg.norm(error)/np.linalg.norm(actual)) if np.linalg.norm(actual)>1e-10 else None,
    step_halving_max_jacobian_change_pu_per_kw=float(np.max(np.abs(sp-sp2))),
    step_halving_max_predicted_delta_change_pu=float(np.max(np.abs((sp-sp2)@dp))),
    reactive_step_halving_max_change_pu_per_kvar=float(np.max(np.abs(sq-sq2))),
    dominant_sender_node=int(np.argmax(np.abs(contrib).sum(axis=0))),dominant_receiver_node=int(np.argmax(np.abs(actual))),
    phase_receiver_sender_mean_voltage_contribution_pu=phase_signed.tolist()))
  matrices.append(sp);qmat.append(sq);vbase.append(v0);vtarget.append(v1);linear.append(approx);deltaP.append(dp)
  print(json.dumps({"case":len(rows),"label":label,"max_error_pu":rows[-1]["approximation_max_abs_error_pu"]}),flush=True)
 mainfiles=[ROOT/"results/network_replay/main_physical"/f"{m}_seed11_panel0_b55_all_tou_eulv_src1p05_fixed_test_physical.npz" for m in ["msecont","dfl"]]
 with np.load(mainfiles[0]) as z:v0=z["household_voltage_pu"].reshape(D,48,55).astype(float)
 with np.load(mainfiles[1]) as z:v1=z["household_voltage_pu"].reshape(D,48,55).astype(float)
 with np.load(a["path"]) as x, np.load(b["path"]) as y:
  hi=[int(np.flatnonzero(x["household_id"]==h)[0]) for h in a["household_id"]]
  forecast_delta=(y["forecast_pv_kw"][hi]-x["forecast_pv_kw"][hi]).transpose(1,2,0)
 actiondelta=(bb-ba).reshape(D,48,55)
 months=np.array([str(d)[:7] for d in dates]);allmonths=np.unique(months)
 np.savez_compressed(OUT/"arrays.npz",case_date=np.array([dates[t//48] for t,_ in selected]),
   case_slot=np.array([t%48 for t,_ in selected]),case_label=np.array([label for _,label in selected]),
   sensitivity_dv_dp=np.array(matrices),sensitivity_dv_dq=np.array(qmat),
   baseline_voltage_pu=vbase,target_voltage_pu=vtarget,linear_voltage_delta_pu=linear,delta_demand_kw=deltaP,
   household_id=a["household_id"],house_phase=net.house_phases,house_bus=np.array(net.house_buses),
   slot=np.arange(48),hour_forecast_pv_delta_kw=forecast_delta[clean].mean(axis=0),
   hour_battery_net_demand_delta_kw=actiondelta[clean].mean(axis=0),
   hour_actual_voltage_delta_pu=(v1-v0)[clean].mean(axis=0),
   hour_voltage_delta_abs_p95_pu=np.quantile(np.abs((v1-v0)[clean]),.95,axis=0),
   date=dates,clean_day=clean,daily_battery_net_demand_delta_kw=np.where(clean[:,None,None],actiondelta,np.nan),
   months=allmonths,monthly_hour_battery_delta_kw=np.array([actiondelta[clean&(months==m)].mean(axis=0) for m in allmonths]),
   monthly_hour_voltage_delta_pu=np.array([(v1-v0)[clean&(months==m)].mean(axis=0) for m in allmonths]))
 result=dict(status="completed",plan=plan,cases=rows,
   max_approximation_error_pu=max(r["approximation_max_abs_error_pu"] for r in rows),
   max_step_halving_predicted_delta_change_pu=max(r["step_halving_max_predicted_delta_change_pu"] for r in rows),
   source_sha256=[sha256(a["path"]),sha256(b["path"])],code_sha256=sha256(__file__),
   boundary="Quasi-static half-hourly constructed-feeder attribution, not dynamic events or a causal weather mechanism.")
 save_json(OUT/"summary.json",result)
if __name__=="__main__":main()

