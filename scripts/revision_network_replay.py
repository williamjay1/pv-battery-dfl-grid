"""Revision-only AC replay; original artifacts are read-only inputs."""
from __future__ import annotations
import argparse, concurrent.futures, json, os, sys, traceback
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(k,"1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE","1")
import network_replay as legacy
from network_model import EuropeanLV
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"revision_20261009/network_replay"
def run(task):
    legacy.OUTPUT=OUT
    legacy.STATUS=OUT/"unused_legacy_status.json"
    return legacy.run_panel(task)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--suite",choices=["matched_external","matched_mapping"],required=True)
    p.add_argument("--workers",type=int,default=3)
    a=p.parse_args()
    common=dict(objectives=["msecont"],seeds=[11],panels=[0],battery_counts=[55],
      tariff_mixes=["all_tou"],period="test",dispatch_suffix="physical",network="eulv",
      source_pu=1.05,power_factor=.95,mapping_seed=None,seasonal_weeks=False,workers=a.workers,
      run_tag=a.suite,suite="revision",overwrite=False)
    if a.suite=="matched_external":
      tasks=[(0,dict(common,network=n,source_pu=s)) for n,s in [
        ("eulv",1.0),("eulv",1.025),("simbench_rural1",1.025),("simbench_semiurb4",1.025)]]
    else: tasks=[(0,dict(common,mapping_seed=i)) for i in range(10)]
    status=dict(status="running",started=legacy.utcnow(),command=sys.argv,
                comparison="continued MSE primary; existing DFL/frozen MSE retained",jobs=[])
    path=OUT/(a.suite+"_status.json")
    legacy.save_json(path,status)
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
      futs={pool.submit(run,t):t for t in tasks}
      for f in concurrent.futures.as_completed(futs):
        try: status["jobs"].append(dict(task=futs[f],status="completed",results=f.result()))
        except Exception as e: status["jobs"].append(dict(task=futs[f],status="failed",error=str(e),traceback=traceback.format_exc()))
        legacy.save_json(path,status)
    status["status"]="completed" if all(j["status"]=="completed" and all(r.get("status")=="completed" for r in j["results"]) for j in status["jobs"]) else "failed"
    status["finished"]=legacy.utcnow();legacy.save_json(path,status)
    print(json.dumps({"status":status["status"],"file":str(path)}),flush=True)
    if status["status"]!="completed": raise SystemExit(1)
if __name__=="__main__": main()

