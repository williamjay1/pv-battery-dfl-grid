"""Supplementary single-postcode replay with actual households only.

The largest postcode group is selected from static metadata, never outcomes.
Occupied customers are assigned in ascending ID order to the first benchmark
load points. Unoccupied points have exactly zero injections. Original feeder
impedances, phases and the 800 kVA transformer are retained. A postcode does not
identify a real compact feeder; this is a geographically narrower condition.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import numpy as np
from battery_model import Battery, DispatchQP, tariff, realized_bill
from network_model import EuropeanLV
from network_replay import save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/geographic_replay"
SCRIPT_SOURCE = Path(__file__).read_bytes()
SCRIPT_SHA256 = hashlib.sha256(SCRIPT_SOURCE).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def select_cohort(args):
    path = ROOT / "datasets/household_postcodes.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    postcodes = metadata["by_household_id"]
    counts = Counter(postcodes.values())
    largest = min(counts, key=lambda p: (-counts[p], str(p)))
    if largest != args.postcode:
        raise ValueError(f"Requested postcode {args.postcode} is not metadata-largest {largest}")
    original = sorted(int(h) for h, postcode in postcodes.items() if postcode == largest)
    splits_path = ROOT / "datasets/splits_v1.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    excluded = sorted(set(original).intersection(splits["excluded_ids"]))
    ids = [h for h in original if h not in excluded] if args.cohort_rule == "protocol_eligible" else original
    if not 1 <= len(ids) <= 55:
        raise ValueError("Cohort must contain 1..55 actual households")
    return np.asarray(ids), {"postcode": largest, "selection": "largest static postcode group; ties lexical; ascending household IDs",
                            "original_household_ids": original, "protocol_excluded_in_original_group": excluded,
                            "cohort_rule": args.cohort_rule, "household_ids": ids,
                            "metadata_sha256": sha256(path), "split_sha256": sha256(splits_path),
                            "metadata_provenance": metadata["provenance"]}


def read_inputs(model, ids, seed, *, decision_arrays="datasets/decision_arrays_v1.npz",
                msecont_arrays="datasets/mse_continuation_arrays_v1.npz", night_zero=False):
    base_path = ROOT / "datasets/forecast_arrays_v1.npz"
    with np.load(base_path) as a:
        hs = np.asarray([int(np.flatnonzero(a["household_id"] == h)[0]) for h in ids])
        ds = np.flatnonzero(a["split_code"] == 2)
        dates = a["date"][ds]
        clean = (a["valid_history"][hs][:, ds] & a["valid_target_feeder"][hs][:, ds])
        finite = a["valid_input_finite"][hs][:, ds]
        actual = {key: a[key][hs][:, ds] for key in ("truth_gc_kw", "truth_pv_kw", "truth_cl_kw")}
        load_forecast = a["gc_shared"][hs][:, ds]
        if model == "mlp":
            forecast = a[f"pv_mlp_seed{seed}"][hs][:, ds]
            forecast_path = base_path
        else:
            filename, key = ((decision_arrays, f"pv_dfl_tou_seed{seed}") if model == "dfl"
                             else (msecont_arrays, f"pv_msecont_tou_seed{seed}"))
            forecast_path = ROOT / filename
            with np.load(forecast_path) as other:
                if not np.array_equal(other["household_id"], a["household_id"]) or not np.array_equal(other["date"], a["date"]):
                    raise ValueError("Forecast household/date axis mismatch")
                forecast = other[key][hs][:, ds]
    if night_zero:
        forecast = forecast.copy()
        forecast[:,:,:8] = 0
        forecast[:,:,42:] = 0
    paths = list(dict.fromkeys([base_path, forecast_path]))
    return dates, actual, load_forecast, forecast, clean, finite, [{"file": str(p), "sha256": sha256(p)} for p in paths]


def formal_overlap_audit(model, seed, ids, dates, charge, discharge, dispatch_suffix):
    path = ROOT / "results/dispatch" / f"{model}_tou_seed{seed}_test_{dispatch_suffix}.npz"
    if not path.exists() or not path.with_suffix(".json").exists():
        return {"status": "not_available", "file": str(path)}
    with np.load(path) as a:
        if not np.array_equal(a["date"], dates):
            raise ValueError("Formal dispatch date axis mismatch")
        lookup = {int(h): i for i, h in enumerate(a["household_id"])}
        ours = np.asarray([i for i, h in enumerate(ids) if int(h) in lookup])
        other = np.asarray([lookup[int(ids[i])] for i in ours])
        c0, d0 = a["charge_kw"][other], a["discharge_kw"][other]
        if not np.array_equal(np.isfinite(c0), np.isfinite(charge[ours])):
            raise ValueError("Formal dispatch finite mask mismatch")
        difference = max(float(np.nanmax(np.abs(c0-charge[ours]))), float(np.nanmax(np.abs(d0-discharge[ours]))))
    if difference > 1e-5:
        raise ValueError(f"Geographic/formal dispatch inconsistency: {difference} kW")
    return {"status": "passed", "n_overlapping_households": len(ours), "max_action_difference_kw": difference,
            "file": str(path), "sha256": sha256(path)}


def replay(model, args, ids, cohort):
    name = f"{model}_seed{args.seed}_postcode{args.postcode}_{args.cohort_rule}_b{len(ids)}_all_tou_test_{args.dispatch_suffix}"
    output = OUTPUT / args.run_tag / (name + ".npz")
    status_path = output.with_suffix(".json")
    dates, actual, load_forecast, forecast, clean, finite, sources = read_inputs(
        model, ids, args.seed, decision_arrays=args.decision_arrays,
        msecont_arrays=args.msecont_arrays, night_zero=args.night_zero)
    code_paths = [ROOT / "scripts/network_model.py", ROOT / "scripts/battery_model.py"]
    provenance = {"sources": sources, "cohort": cohort, "source_pu": args.source_pu,
                  "power_factor": args.power_factor, "seed": args.seed, "night_zero": args.night_zero,
                  "physical_code": [{"file": str(p), "sha256": sha256(p)} for p in code_paths]}
    if output.exists() and status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            if previous["provenance"] != provenance:
                raise ValueError("Completed geographic case inputs changed; choose a new run tag")
            print(json.dumps({"scenario": name, "status": "already_completed"}), flush=True)
            return previous
    status = {"scenario": name, "objective": model, "status": "running", "started_utc": now(),
              "provenance": provenance, "output": str(output), "parameters": vars(args),
              "script_sha256": SCRIPT_SHA256, "script_hash_semantics": "import_snapshot",
              "boundary": "Postcode does not establish a real compact feeder. Only actual occupied household nodes contribute to customer metrics and bill denominators; the original 55-point network and 800 kVA transformer remain unchanged."}
    save_json(status_path, status)
    try:
        start = time.perf_counter()
        battery = Battery()
        buy, sell = tariff("tou")
        qp = DispatchQP(battery, "tou", tolerance=1e-13)
        gc, pv, cl = [actual[k] for k in ("truth_gc_kw", "truth_pv_kw", "truth_cl_kw")]
        charge = np.full_like(gc, np.nan)
        discharge = np.full_like(gc, np.nan)
        energy = np.full_like(gc, np.nan)
        bills = np.full(gc.shape[:2], np.nan)
        failures, max_balance, max_simultaneous, max_constraint = [], 0., 0., 0.
        for h in range(len(ids)):
            for d in range(len(dates)):
                if not finite[h,d] or not np.isfinite(forecast[h,d]).all() or not np.isfinite(load_forecast[h,d]).all():
                    continue
                try:
                    c, dc, e, diag = qp.solve(load_forecast[h,d]-forecast[h,d], True)
                    charge[h,d], discharge[h,d], energy[h,d] = c, dc, e
                    bills[h,d] = realized_bill(gc[h,d]-pv[h,d], c, dc, buy, sell, battery) + .5*.18*np.sum(cl[h,d])
                    max_balance = max(max_balance, diag["energy_balance_max"])
                    max_simultaneous = max(max_simultaneous, diag["max_simultaneous_kw"])
                    max_constraint = max(max_constraint, diag["constraint_violation"])
                except Exception as exc:
                    failures.append({"household_id": int(ids[h]), "date": str(dates[d]), "error": str(exc)})
        if failures:
            status["dispatch_failures"] = failures
            raise RuntimeError(f"{len(failures)} dispatch failures retained")
        overlap = formal_overlap_audit(model, args.seed, ids, dates, charge, discharge, args.dispatch_suffix)
        T, H = len(dates)*48, len(ids)
        def pad(x):
            padded = np.zeros((T,55))
            padded[:,:H] = x.transpose(1,2,0).reshape(T,H)
            return padded
        gross, solar, action = pad(gc+cl), pad(pv), pad(charge-discharge)
        network = EuropeanLV(source_pu=args.source_pu)
        result = network.run(gross, solar, action, power_factor=args.power_factor, return_bus=True)
        voltage = result["household_voltage_pu"][:,:H]
        arrays = {key: value for key,value in result.items() if isinstance(value,np.ndarray)}
        arrays["all_load_point_voltage_pu"] = result["household_voltage_pu"].astype(np.float32)
        arrays["household_voltage_pu"] = voltage.astype(np.float32)
        arrays["vmin_pu"] = voltage.min(axis=1)
        arrays["vmax_pu"] = voltage.max(axis=1)
        for suffix, lower, upper in [("", .95,1.05),("_090_110",.90,1.10)]:
            arrays["voltage_violation_fraction"+suffix] = ((voltage<lower)|(voltage>upper)).mean(axis=1)
            arrays["voltage_violation_fraction"+suffix][~result["converged"]] = np.nan
            arrays["voltage_exceedance_pu"+suffix] = (np.maximum(lower-voltage,0)+np.maximum(voltage-upper,0)).sum(axis=1)
            arrays["voltage_exceedance_mean_pu"+suffix] = arrays["voltage_exceedance_pu"+suffix]/H
        complete = result["input_valid"].reshape(-1,48).all(axis=1)
        common_clean = clean.all(axis=0)
        if np.any(common_clean & ~complete):
            raise ValueError("A source-clean date lacks physical inputs or dispatch; no method-specific dropping allowed")
        arrays.update(date=dates, date_slot=np.repeat(dates,48), household_id=ids,
                      benchmark_load_index=np.arange(H), occupied_load_point=np.arange(55)<H,
                      voltage_household_id=ids, clean_household_day=clean, clean_day=common_clean,
                      physical_input_complete_day=complete, converged_day=result["converged"].reshape(-1,48).all(axis=1),
                      charge_kw=charge, discharge_kw=discharge, energy_kwh=energy,
                      actual_load_kw=gc+cl, actual_pv_kw=pv,
                      bill_byhouse_aud=bills.T, bill_mean_aud=bills.mean(axis=0),
                      power_balance_residual_kw=result["source_import_kw"]-(gross-solar+action).sum(axis=1)-result["loss_kw"])
        temporary = output.with_name(output.stem+f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
        status.update(status="completed", finished_utc=now(), elapsed_seconds=time.perf_counter()-start,
                      n_households=H, n_dates=len(dates), n_clean_dates=int(common_clean.sum()),
                      n_physical_complete_days=int(complete.sum()), n_converged_slots=int(result["converged"].sum()),
                      n_solver_failed_slots=int((result["input_valid"] & ~result["converged"]).sum()),
                      clean_dates=dates[common_clean].tolist(), battery=battery.__dict__, solver_tolerance=1e-13,
                      max_energy_balance_residual_kwh=max_balance, max_simultaneous_kw=max_simultaneous,
                      max_constraint_violation=max_constraint, formal_overlap_audit=overlap,
                      max_power_balance_residual_kw=float(np.nanmax(np.abs(arrays["power_balance_residual_kw"]))),
                      full_365_day_physical_coverage=bool(len(dates)==365 and result["converged"].all()),
                      clean_day_definition="valid_history & valid_target_feeder, intersected over the exact occupied cohort; independent of model outputs",
                      network_metadata=result["metadata"], output_sha256=sha256(output), test_used_for_fitting_or_selection=False)
        save_json(status_path,status)
        print(json.dumps({k:status[k] for k in ("scenario","status","n_households","n_clean_dates","n_physical_complete_days","n_solver_failed_slots","elapsed_seconds")}),flush=True)
        return status
    except Exception as exc:
        status.update(status="failed", finished_utc=now(), error=str(exc), traceback=traceback.format_exc())
        save_json(status_path,status)
        raise


def summarize_geography(args):
    """Paired calendar-block analysis with the actual occupied denominator."""
    from analyze_network import bootstrap_indices, summarize
    folder = OUTPUT / args.run_tag
    source = {}
    metadata = {}
    for path in sorted(folder.glob("*.npz")):
        if not path.with_suffix(".json").exists():
            continue
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if report.get("status") != "completed" or report.get("objective") not in ["mlp","dfl","msecont"]:
            continue
        model = report["objective"]
        if model in source:
            raise ValueError("Multiple cases for one model; use a separate run tag")
        if report["n_solver_failed_slots"]:
            raise ValueError("Geographic solver failures require review")
        if report["provenance"]["seed"] != args.seed or report["parameters"]["dispatch_suffix"] != args.dispatch_suffix:
            raise ValueError("Geographic seed or dispatch variant mismatch")
        if sha256(path) != report["output_sha256"]:
            raise ValueError("Geographic output hash mismatch")
        metadata[model] = report
        with np.load(path) as a:
            source[model] = {k:a[k] for k in a.files}
    if set(source) != {"mlp","dfl","msecont"}:
        raise ValueError("Geographic summary requires all three completed methods")
    first = source["mlp"]
    dates, ids = first["date"], first["household_id"]
    H, D = len(ids), len(dates)
    if H != 27 or 2 in ids or D != 365 or not np.all(np.diff(dates.astype("datetime64[D]")).astype(int)==1):
        raise ValueError("Expected protocol-eligible 27-household, full-calendar cohort")
    for model, a in source.items():
        for key in ["date","household_id","clean_day","converged_day","occupied_load_point"]:
            if not np.array_equal(first[key],a[key]):
                raise ValueError(f"Unmatched paired geographic field {model}/{key}")
        if a["household_voltage_pu"].shape != (D*48,H) or a["bill_byhouse_aud"].shape != (D,H):
            raise ValueError("Customer metric/bill dimension mismatch")
        if not np.array_equal(a["bill_byhouse_aud"].mean(axis=1),a["bill_mean_aud"]):
            raise ValueError("Incorrect occupied-household bill denominator")
        if not np.allclose(a["voltage_exceedance_mean_pu"],a["voltage_exceedance_pu"]/H,atol=0,rtol=0,equal_nan=True):
            raise ValueError("Incorrect occupied-household voltage denominator")
    clean = first["clean_day"] & first["converged_day"]
    all_metrics = {}
    for model,a in source.items():
        slots = lambda key:a[key].reshape(D,48)
        imp = slots("source_import_kw")
        metrics = {"bill_aud":a["bill_mean_aud"].astype(float).copy(),
                   "voltage_percentage_points":100*slots("voltage_violation_fraction").mean(axis=1),
                   "mean_voltage_excursion_pu":slots("voltage_exceedance_mean_pu").mean(axis=1),
                   "daily_import_peak_kw":np.maximum(imp,0).max(axis=1),
                   "daily_export_peak_kw":np.maximum(-imp,0).max(axis=1),
                   "loss_kwh":.5*slots("loss_kw").sum(axis=1),
                   "voltage_090_110_percentage_points":100*slots("voltage_violation_fraction_090_110").mean(axis=1)}
        for value in metrics.values():
            if not np.isfinite(value[clean]).all():
                raise ValueError("Nonfinite primary-clean geographic metric")
            value[~clean] = np.nan
        all_metrics[model] = metrics
    indices = {b:bootstrap_indices(D,b,args.bootstrap_reps,20261005) for b in [7,14,28]}
    comparisons = {}
    daily = {"date":dates,"common_clean_day":clean,"household_count":np.asarray(H)}
    for model,metrics in all_metrics.items():
        daily.update({f"{model}__{key}":value for key,value in metrics.items()})
    for baseline in ["mlp","msecont"]:
        comparisons[baseline] = {}
        for metric in all_metrics[baseline]:
            x = all_metrics["dfl"][metric]-all_metrics[baseline][metric]
            adjacent = np.isfinite(x[:-1]) & np.isfinite(x[1:])
            lag1 = (float(np.corrcoef(x[:-1][adjacent],x[1:][adjacent])[0,1])
                    if adjacent.sum()>2 and np.std(x[:-1][adjacent])>0 and np.std(x[1:][adjacent])>0 else None)
            comparisons[baseline][metric] = {
                "baseline_mean":float(np.nanmean(all_metrics[baseline][metric])),
                "dfl_mean":float(np.nanmean(all_metrics["dfl"][metric])),
                "delta_dfl_minus_baseline":summarize(x,indices[7]),
                "block_sensitivity":{str(b):summarize(x,indices[b])["ci95"] for b in [14,28]},
                "lag1_correlation_of_adjacent_clean_daily_differences":lag1}
            daily[f"dfl_minus_{baseline}__{metric}"] = x
    summary = {"run_tag":args.run_tag,"postcode":"2259","n_households":H,"seed":args.seed,
               "n_calendar_dates":D,"n_common_clean_dates":int(clean.sum()),
               "n_common_physical_complete_dates":int(first["converged_day"].sum()),
               "customer_metric_denominator":H,"battery_owner_cost_denominator":H,
               "inference":"Paired circular calendar-block percentile intervals on the full 365-date axis with common non-clean dates retained as missing; 7-day primary blocks and 14/28-day sensitivity, 2000 replicates by default. Fixed postcode cohort, fixed benchmark mapping, one training seed. No iid half-hour or household inference; intervals do not represent training-seed variability or real-feeder population uncertainty.",
               "assumptions":"Time blocks accommodate short-range dependence; normality and equal group variances are not required. Seasonal and longer-range dependence remain a limitation, assessed only through fixed block-length sensitivity, not optimized against outcomes.",
               "geographic_boundary":"Postcode membership does not demonstrate a compact physical feeder; unused benchmark load points remain zero-injection points, not invented households.",
               "bootstrap_replicates":args.bootstrap_reps,"bootstrap_seed":20261005,
               "analysis_script_sha256":SCRIPT_SHA256,
               "bootstrap_helper_sha256":sha256(ROOT/"scripts/analyze_network.py"),
               "models":{m:{k:float(np.nanmean(v)) for k,v in metrics.items()} for m,metrics in all_metrics.items()},
               "comparisons":comparisons,
               "provenance":{m:{"output":r["output"],"output_sha256":r["output_sha256"],"cohort":r["provenance"]["cohort"]} for m,r in metadata.items()}}
    np.savez_compressed(folder/"daily.npz",**daily)
    columns = [key for key,value in daily.items() if np.ndim(value)==1 and key!="date"]
    with (folder/"daily.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["date","household_count"]+columns)
        for i,day in enumerate(dates):
            writer.writerow([str(day),H]+[daily[key][i].item() for key in columns])
    save_json(folder/"summary.json",summary)
    print(json.dumps({"geographic_summary":"completed","run_tag":args.run_tag,"n_common_clean_dates":int(clean.sum()),
                      "cost_deltas":{base:comparisons[base]["bill_aud"]["delta_dfl_minus_baseline"] for base in comparisons}}),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objectives",nargs="+",choices=["mlp","dfl","msecont"],default=["mlp","dfl","msecont"])
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--postcode",default="2259")
    ap.add_argument("--cohort-rule",required=True,choices=["original","protocol_eligible"])
    ap.add_argument("--source-pu",type=float,default=1.05)
    ap.add_argument("--power-factor",type=float,default=.95)
    ap.add_argument("--run-tag",default="geographic_v1")
    ap.add_argument("--decision-arrays",default="datasets/decision_arrays_v1.npz")
    ap.add_argument("--msecont-arrays",default="datasets/mse_continuation_arrays_v1.npz")
    ap.add_argument("--night-zero",action="store_true")
    ap.add_argument("--dispatch-suffix",default="tight")
    ap.add_argument("--summarize-only",action="store_true")
    ap.add_argument("--bootstrap-reps",type=int,default=2000)
    args=ap.parse_args()
    if Path(args.run_tag).name!=args.run_tag:raise ValueError("Simple run tag required")
    destination=OUTPUT/args.run_tag
    destination.mkdir(parents=True,exist_ok=True)
    (destination/("geographic_replay_"+SCRIPT_SHA256+".py")).write_bytes(SCRIPT_SOURCE)
    if args.summarize_only:
        summarize_geography(args)
        return
    ids,cohort=select_cohort(args)
    records=[replay(model,args,ids,cohort) for model in args.objectives]
    save_json(destination/"manifest.json",{"status":"completed","cohort":cohort,"scenarios":records,"updated_utc":now()})


if __name__=="__main__":
    main()
