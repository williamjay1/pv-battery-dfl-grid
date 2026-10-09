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
    args=ap.parse_args()
    if Path(args.run_tag).name!=args.run_tag:raise ValueError("Simple run tag required")
    destination=OUTPUT/args.run_tag
    destination.mkdir(parents=True,exist_ok=True)
    (destination/("geographic_replay_"+SCRIPT_SHA256+".py")).write_bytes(SCRIPT_SOURCE)
    ids,cohort=select_cohort(args)
    records=[replay(model,args,ids,cohort) for model in args.objectives]
    save_json(destination/"manifest.json",{"status":"completed","cohort":cohort,"scenarios":records,"updated_utc":now()})


if __name__=="__main__":
    main()
