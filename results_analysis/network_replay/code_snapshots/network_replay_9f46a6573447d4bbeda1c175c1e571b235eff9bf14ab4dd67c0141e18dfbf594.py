"""Frozen-scenario AC replay of completed dispatch files.

No outcomes are used to choose homes, mappings, adoption, tariffs, or dates.
CLI defaults implement the 36 cells per forecasting objective in the protocol.
Dispatch JSON acts as a completion marker; missing inputs remain pending. Every
scenario has its own status file, and errors are retained rather than omitted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
from network_model import EuropeanLV, SimBenchLV

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "results/dispatch"
OUTPUT = ROOT / "results/network_replay"
STATUS = ROOT / "research/network_replay_status.json"
REPLAY_SCRIPT_SOURCE = Path(__file__).read_bytes()
REPLAY_SCRIPT_SHA256 = hashlib.sha256(REPLAY_SCRIPT_SOURCE).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(2 ** 20):
            digest.update(block)
    return digest.hexdigest()


def write_manifest():
    records = []
    keys = ["scenario", "status", "objective", "seed", "panel", "battery_count", "tariff_mix",
            "n_dates", "n_clean_dates", "input_valid_slots", "converged_slots", "solver_failed_slots",
            "missing_input_slots", "full_365_day_physical_coverage", "output_sha256", "elapsed_seconds"]
    for path in sorted(OUTPUT.glob("*/*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "scenario" not in value:
            continue
        row = {k: value.get(k) for k in keys}
        row.update(metadata_file=str(path), output_file=value.get("output"),
                   run_tag=path.parent.name, parameters=value.get("parameters", {}))
        records.append(row)
    manifest = {"updated_utc": utcnow(), "scenarios": records,
                "complete_scenarios": sum(r["status"] == "completed" for r in records),
                "driver_hash_note": "Early replay hashes describe the driver file at scenario start; metadata guards were strengthened during initial MSE runs without changing network physics or numerical operations. Later runs additionally record import_snapshot semantics and preserve the exact driver source in code_snapshots. The network_model_sha256 and dispatch input hashes are frozen for every case.",
                "coverage_expansion_note": "Originally fixed four weeks retained as prespecified checks. Full-year-axis external/source/mapping replay added because panel0 has only 11 common clean days in the original windows; expansion uses all available dates and was authorized without choosing on network effects."}
    save_json(OUTPUT / "scenario_manifest.json", manifest)


def seasonal_week_dates(dates):
    """First complete Monday-Sunday for each Australian meteorological season.

    On the frozen July-2012 to June-2013 axis this yields weeks beginning
    2012-07-02, 2012-09-03, 2012-12-03, 2013-03-04, independent of outcomes.
    """
    dates = sorted(date.fromisoformat(str(x)) for x in dates)
    available = set(dates)
    seasons = {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
               6: "winter", 7: "winter", 8: "winter", 9: "spring", 10: "spring", 11: "spring"}
    chosen, seen = [], set()
    for day in dates:
        season = seasons[day.month]
        if day.weekday() != 0 or season in seen:
            continue
        week = [day + timedelta(days=j) for j in range(7)]
        if all(x in available and seasons[x.month] == season for x in week):
            seen.add(season)
            chosen.extend(str(x) for x in week)
    if len(seen) != 4:
        raise ValueError("Four complete meteorological-season weeks are not available")
    return chosen


def scenario_name(objective, seed, panel, adoption, tariff_mix, args):
    mapping = "fixed" if args["mapping_seed"] is None else f"map{args['mapping_seed']}"
    period = "seasonal" if args["seasonal_weeks"] else args["period"]
    source = str(args["source_pu"]).replace(".", "p")
    return f"{objective}_seed{seed}_panel{panel}_b{adoption}_{tariff_mix}_{args['network']}_src{source}_{mapping}_{period}_{args['dispatch_suffix']}"


def source_path(model, tariff, seed, args):
    return DISPATCH / f"{model}_{tariff}_seed{seed}_{args['period']}_{args['dispatch_suffix']}.npz"


def load_source(model, tariff, seed, ids, args):
    path = source_path(model, tariff, seed, args)
    report_path = path.with_suffix(".json")
    if not path.exists() or not report_path.exists():
        raise FileNotFoundError(f"Awaiting completed dispatch: {path.name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures = max(int(report.get("n_failed", 0)), int(report.get("failed", 0)))
    if failures:
        raise ValueError(f"Dispatch reports {failures} failures: {path.name}")
    reported_model = report.get("model", report.get("method"))
    if reported_model != model or report.get("tariff") != tariff or report.get("seed") != seed:
        raise ValueError(f"Dispatch metadata mismatch: {path.name}")
    if report.get("test_used_for_fitting", True):
        raise ValueError(f"Dispatch does not attest no test fitting: {path.name}")
    if args["dispatch_suffix"] == "tight" and report.get("solver_tolerance") != 1e-13:
        raise ValueError(f"Formal tight dispatch must attest solver tolerance 1e-13: {path.name}")
    required = ("charge_kw", "discharge_kw", "bill_aud", "clean_day", "actual_load_kw", "actual_pv_kw")
    with np.load(path) as data:
        if len(set(data["household_id"].tolist())) != len(data["household_id"]):
            raise ValueError("Duplicate household IDs in dispatch")
        lookup = {int(x): i for i, x in enumerate(data["household_id"])}
        order = np.asarray([lookup[int(h)] for h in ids], dtype=int)
        values = {key: np.asarray(data[key][order]).copy() for key in required}
        values["date"] = data["date"].copy()
    if values["actual_load_kw"].shape != (55, len(values["date"]), 48):
        raise ValueError("Expected 55 x dates x 48 dispatch input")
    values["provenance"] = {"file": str(path), "sha256": sha256(path), "report": report}
    return values


def same_arrays(a, b, name):
    if not np.array_equal(a, b, equal_nan=True):
        raise ValueError(f"Unmatched paired source field: {name}")


def assemble_sources(learned_tou, learned_flat, none_tou, none_flat, adoption, tariff_mix):
    sources = [learned_tou, learned_flat, none_tou, none_flat]
    for source in sources[1:]:
        if not np.array_equal(sources[0]["date"], source["date"]):
            raise ValueError("Unmatched dates across tariff/no-battery arms")
        for key in ("actual_load_kw", "actual_pv_kw", "clean_day"):
            same_arrays(sources[0][key], source[key], key)
    use_tou = np.ones(55, bool)
    if tariff_mix == "mixed_first28_tou_rest_flat":
        use_tou[28:] = False
    installed = np.arange(55) < adoption
    battery = np.zeros_like(learned_tou["charge_kw"])
    bill = np.full_like(learned_tou["bill_aud"], np.nan)
    for h in range(55):
        source = sources[(0 if use_tou[h] else 1) if installed[h] else (2 if use_tou[h] else 3)]
        if installed[h]:
            battery[h] = source["charge_kw"][h] - source["discharge_kw"][h]
        bill[h] = source["bill_aud"][h]
    return {"load_kw": learned_tou["actual_load_kw"], "pv_kw": learned_tou["actual_pv_kw"],
            "battery_kw": battery, "bill_byhouse_aud": bill.T,
            "none_bill_tou_aud": none_tou["bill_aud"].T,
            "none_bill_flat_aud": none_flat["bill_aud"].T,
            "clean_day": np.logical_and.reduce([s["clean_day"].all(axis=0) for s in sources]),
            "installed": installed, "use_tou": use_tou, "date": learned_tou["date"]}


def replay_one(model, source, ids, objective, seed, panel_index, adoption, tariff_mix, args,
               expected_clean_dates, source_provenance, frozen_splits_hash):
    name = scenario_name(objective, seed, panel_index, adoption, tariff_mix, args)
    destination = OUTPUT / args["run_tag"]
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / (name + ".json")
    output_path = destination / (name + ".npz")
    status = {"scenario": name, "status": "running", "started_utc": utcnow(),
              "objective": objective, "seed": seed, "panel": panel_index,
              "battery_count": adoption, "tariff_mix": tariff_mix, "parameters": args,
              "household_id": list(map(int, ids)), "split_sha256": frozen_splits_hash,
              "network_model_sha256": sha256(ROOT / "scripts/network_model.py"),
              "replay_script_sha256": REPLAY_SCRIPT_SHA256,
              "replay_script_hash_semantics": "import_snapshot",
              "sources": source_provenance, "output": str(output_path)}
    if status_path.exists() and output_path.exists() and not args["overwrite"]:
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "completed":
            previous = [(x["file"], x["sha256"]) for x in old["sources"]]
            current = [(x["file"], x["sha256"]) for x in source_provenance]
            physics_keys = ["network", "source_pu", "power_factor", "mapping_seed", "seasonal_weeks", "period", "dispatch_suffix"]
            changed_parameters = any(old["parameters"][k] != args[k] for k in physics_keys)
            changed_model = old.get("network_model_sha256") != status["network_model_sha256"]
            if previous != current or old["split_sha256"] != frozen_splits_hash or changed_parameters or changed_model:
                raise ValueError(f"Completed scenario input changed: {name}; choose a new run tag")
            print(json.dumps({"scenario": name, "status": "already_completed"}), flush=True)
            return old
    save_json(status_path, status)
    try:
        frozen_clean = np.isin(source["date"], expected_clean_dates)
        same_arrays(frozen_clean, source["clean_day"], "frozen common clean-day mask")
        selected = np.arange(len(source["date"]))
        if args["seasonal_weeks"]:
            selected = np.flatnonzero(np.isin(source["date"], seasonal_week_dates(source["date"])))
        dates = source["date"][selected]
        load, pv, battery = [source[k][:, selected].transpose(1, 2, 0).reshape(-1, 55)
                             for k in ("load_kw", "pv_kw", "battery_kw")]
        permutation = None
        if args["mapping_seed"] is not None:
            permutation = np.random.default_rng(args["mapping_seed"]).permutation(55)
        run_start = time.perf_counter()
        out = model.run(load, pv, battery, mapping=permutation,
                        power_factor=args["power_factor"], voltage_band=(.95, 1.05), return_bus=True)
        arrays = {key: value.astype(np.float32) if key == "household_voltage_pu" else value
                  for key, value in out.items() if isinstance(value, np.ndarray)}
        nominal_net = (load.astype(float) - pv.astype(float) + battery.astype(float)).sum(axis=1)
        arrays["power_balance_residual_kw"] = out["source_import_kw"] - nominal_net - out["loss_kw"]
        arrays.update(date=dates, household_id=np.asarray(ids), slot=np.tile(np.arange(48), len(dates)),
                      date_slot=np.repeat(dates, 48), clean_day=source["clean_day"][selected],
                      battery_installed=source["installed"], tariff_is_tou=source["use_tou"],
                      bill_byhouse_aud=source["bill_byhouse_aud"][selected],
                      bill_mean_aud=source["bill_byhouse_aud"][selected].mean(axis=1),
                      physical_input_complete_day=out["input_valid"].reshape(-1, 48).all(axis=1),
                      converged_day=out["converged"].reshape(-1, 48).all(axis=1),
                      mapping_household_index=np.arange(55) if permutation is None else permutation,
                      voltage_household_id=np.asarray(ids) if permutation is None else np.asarray(ids)[permutation])
        if objective == "none":
            arrays["none_bill_byhouse_tou_aud"] = source["none_bill_tou_aud"][selected]
            arrays["none_bill_byhouse_flat_aud"] = source["none_bill_flat_aud"][selected]
        # These extra thresholds are calculated for all cells, never selected by
        # observed performance. Float64 voltages are used before storage casting.
        vh = out["household_voltage_pu"]
        arrays["voltage_violation_fraction_090_110"] = ((vh < .90) | (vh > 1.10)).mean(axis=1)
        arrays["voltage_exceedance_pu_090_110"] = (np.maximum(.90-vh, 0)+np.maximum(vh-1.10, 0)).sum(axis=1)
        arrays["voltage_violation_fraction_090_110"][~out["converged"]] = np.nan
        temporary = output_path.with_name(output_path.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output_path)
        status.update(status="completed", finished_utc=utcnow(), elapsed_seconds=time.perf_counter()-run_start,
                      network_elapsed_seconds=out["elapsed_seconds"], network_metadata=out["metadata"],
                      n_dates=len(dates), n_clean_dates=int(arrays["clean_day"].sum()),
                      input_valid_slots=int(out["input_valid"].sum()), converged_slots=int(out["converged"].sum()),
                      solver_failed_slots=int((out["input_valid"] & ~out["converged"]).sum()),
                      missing_input_slots=int((~out["input_valid"]).sum()),
                      max_power_balance_residual_kw=float(np.nanmax(np.abs(arrays["power_balance_residual_kw"]))) if out["converged"].any() else None,
                      full_365_day_physical_coverage=bool(len(dates)==365 and out["converged"].all()),
                      full_365_day_clean_coverage=bool(len(dates)==365 and arrays["clean_day"].all()),
                      clean_day_definition="intersection across all 55 households, source flags and frozen split dates; identical across learned arms",
                      output_sha256=sha256(output_path))
        if status["solver_failed_slots"]:
            status["quality_flag"] = "solver_failures_retained_require_review_before_inference"
        save_json(status_path, status)
        print(json.dumps({k: status[k] for k in ("scenario", "status", "n_clean_dates", "converged_slots", "solver_failed_slots", "elapsed_seconds")}), flush=True)
        return status
    except Exception as error:
        status.update(status="failed", finished_utc=utcnow(), error=str(error), traceback=traceback.format_exc())
        save_json(status_path, status)
        raise


def run_panel(task):
    panel_index, args = task
    splits_path = ROOT / "datasets/splits_v1.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    ids = splits["primary_panels"][panel_index] if panel_index < 3 else splits["heldout_network_household_ids"]
    expected = splits["primary_panel_common_clean_dates"][panel_index][args["period"]] if panel_index < 3 else splits["heldout_network_common_clean_dates"][args["period"]]
    model = EuropeanLV(source_pu=args["source_pu"]) if args["network"] == "eulv" else SimBenchLV(args["network"])
    results = []
    none_tou = load_source("none", "tou", 11, ids, args)
    none_flat = load_source("none", "flat", 11, ids, args)
    for objective in args["objectives"]:
        for seed in args["seeds"]:
            try:
                tou = load_source(objective, "tou", seed, ids, args)
                flat = load_source(objective, "flat", seed, ids, args)
            except Exception as error:
                input_status = "pending_inputs" if isinstance(error, FileNotFoundError) else "failed"
                results.append({"panel": panel_index, "objective": objective, "seed": seed,
                                "status": input_status, "message": str(error)})
                for adoption in args["battery_counts"]:
                    for mix in args["tariff_mixes"]:
                        name = scenario_name(objective, seed, panel_index, adoption, mix, args)
                        failure_path = OUTPUT / args["run_tag"] / (name + ".json")
                        if failure_path.exists() and json.loads(failure_path.read_text(encoding="utf-8")).get("status") == "completed":
                            failure_path = failure_path.with_name(name + ".retry_input_failure.json")
                        save_json(failure_path,
                                  {"scenario": name, "status": input_status, "message": str(error),
                                   "objective": objective, "seed": seed, "panel": panel_index,
                                   "battery_count": adoption, "tariff_mix": mix,
                                   "traceback": traceback.format_exc(),
                                   "parameters": args, "updated_utc": utcnow()})
                continue
            provenance = [s["provenance"] for s in (tou, flat, none_tou, none_flat)]
            for adoption in args["battery_counts"]:
                for tariff_mix in args["tariff_mixes"]:
                    source = assemble_sources(tou, flat, none_tou, none_flat, adoption, tariff_mix)
                    results.append(replay_one(model, source, ids, objective, seed, panel_index, adoption,
                                              tariff_mix, args, expected, provenance, sha256(splits_path)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objectives", nargs="+", default=["mlp"],
                        choices=["mlp", "dfl", "none", "hgb", "calibrated", "stochastic", "msecont"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--panels", nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2, 3])
    parser.add_argument("--battery-counts", nargs="+", type=int, default=[28, 55], choices=[0, 28, 55])
    parser.add_argument("--tariff-mixes", nargs="+", default=["all_tou", "mixed_first28_tou_rest_flat"],
                        choices=["all_tou", "mixed_first28_tou_rest_flat"])
    parser.add_argument("--period", default="test", choices=["test", "validation"])
    parser.add_argument("--dispatch-suffix", default="tight")
    parser.add_argument("--network", default="eulv", choices=["eulv", "simbench_rural1", "simbench_semiurb4"])
    parser.add_argument("--source-pu", type=float, default=1.05)
    parser.add_argument("--power-factor", type=float, default=.95)
    parser.add_argument("--mapping-seed", type=int, default=None)
    parser.add_argument("--seasonal-weeks", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run-tag", default="main_v1")
    parser.add_argument("--suite", default="main", choices=["main", "external", "mapping"],
                        help="external: two SimBench and two source-voltage cases; mapping: permutations with seeds 0..9. Both fixed at panel0, seed11, b55, all_tou.")
    parser.add_argument("--overwrite", action="store_true")
    args = vars(parser.parse_args())
    if Path(args["run_tag"]).name != args["run_tag"]:
        raise ValueError("run-tag must be a simple folder name")
    if not 1 <= args["workers"] <= 3:
        raise ValueError("workers must be 1-3")
    if args["network"] != "eulv":
        args["source_pu"] = 1.025
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = OUTPUT / "code_snapshots" / ("network_replay_" + REPLAY_SCRIPT_SHA256 + ".py")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(REPLAY_SCRIPT_SOURCE)
    tasks = [(panel, args) for panel in args["panels"]]
    if args["suite"] != "main":
        common = {**args, "panels": [0], "seeds": [11], "battery_counts": [55], "tariff_mixes": ["all_tou"]}
        if args["suite"] == "external":
            tasks = [(0, {**common, "network": network, "source_pu": source_pu, "mapping_seed": None})
                     for network, source_pu in [("simbench_rural1", 1.025), ("simbench_semiurb4", 1.025), ("eulv", 1.0), ("eulv", 1.025)]]
        else:
            tasks = [(0, {**common, "network": "eulv", "source_pu": 1.05, "mapping_seed": mapping_seed}) for mapping_seed in range(10)]
    status = {"started_utc": utcnow(), "command": sys.argv, "parameters": args, "status": "running", "jobs": {}}
    save_json(STATUS, status)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args["workers"], len(tasks))) as pool:
        futures = {pool.submit(run_panel, task): f"panel{task[0]}_{task[1]['network']}_source{task[1]['source_pu']}_mapping{task[1]['mapping_seed']}" for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                status["jobs"][job] = {"status": "completed", "scenarios": future.result()}
            except Exception as error:
                status["jobs"][job] = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
            save_json(STATUS, status)
    failed = any(p["status"] == "failed" or any(s.get("status") == "failed" for s in p.get("scenarios", []))
                 for p in status["jobs"].values())
    pending = any(s.get("status") == "pending_inputs" for p in status["jobs"].values() for s in p.get("scenarios", []))
    status.update(status="failed" if failed else "pending_inputs" if pending else "completed", finished_utc=utcnow())
    save_json(STATUS, status)
    write_manifest()
    print(json.dumps({"status": status["status"], "jobs": len(status["jobs"]), "status_file": str(STATUS)}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
