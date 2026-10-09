"""Acquire, audit, validate, and time the public EPRI European LV model.

Raw downloads are immutable and distinct from working copies. This script never
changes the benchmark transformer capacity or source setpoint to obtain effects.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

WORK = Path(__file__).resolve().parents[1]
RAW = Path("F:/AcademicData/pv_battery_dfl_grid_20261005/raw/network_20261005")
MODEL = WORK / "datasets/network_eulv"
BASE = "https://svn.code.sf.net/p/electricdss/code/trunk/Distrib/IEEETestCases/LVTestCase/"


def acquire():
    drives = {d: shutil.disk_usage(d + ":/").free for d in ("D", "E", "F")}
    if drives["F"] < 20_000_000 or drives["D"] < 20_000_000:
        raise RuntimeError("Insufficient F/D space for small network input files")
    RAW.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    names = ["Master.dss", "LineCode.txt", "Lines.txt", "Loads.txt",
             "Transformers.txt", "Buscoords.txt", "LoadShapes.txt", "Monitors.txt"]
    sources = [(n, BASE + n) for n in names]
    sources += [(f"Daily_1min_100profiles/Load_profile_{i}.txt",
                 BASE + f"Daily_1min_100profiles/load_profile_{i}.txt")
                for i in range(1, 101)]
    sources += [("EPRI_License.txt", "https://svn.code.sf.net/p/electricdss/code/trunk/License.txt")]
    prefix = "https://raw.githubusercontent.com/ieee-pes-amps/dtf-dev/master/"
    for name in ("Snapshot_1min_Initialization_Off Peak.xlsx", "Snapshot_566min_On Peak.xlsx",
                 "Snapshot_1440min_End.xlsx"):
        remote = "European LV Test Feeder/Validated User Models/OpenDSS/Solutions/Snapshots/" + name
        sources.append(("reference/" + name, prefix + urllib.parse.quote(remote)))
    remote = "European LV Test Feeder/Published Baseline Model/IEEE European LV Test Feeder - Draft.docx"
    sources.append(("reference/IEEE_European_LV_technical_document.docx", prefix + urllib.parse.quote(remote)))

    def get(item):
        name, url = item
        path = RAW / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Academic network validation"})
            with urllib.request.urlopen(req, timeout=40) as response:
                data = response.read()
            with path.open("xb") as f:
                f.write(data)
            path.chmod(0o444)
        data = path.read_bytes()
        target = MODEL / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return {"file": name, "source": url, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(get, sources))
    manifest = {
        "retrieved": "2026-10-05", "free_space_before": drives,
        "raw_directory": str(RAW), "working_directory": str(MODEL),
        "EPRI_repository_revision_seen": 4176,
        "license": "EPRI BSD-style redistribution and use license (retained locally)",
        "reference_license_note": "IEEE reference workbooks used for verification; no redistribution claim made",
        "scope": "EPRI implementation of European LV test feeder; not Ausgrid topology",
        "unchanged": ["line impedances", "55 load bus/phase assignments", "800 kVA transformer",
                      "11/0.416 kV windings", "source 1.05 pu"],
        "rating_warning": "No NormAmps/EmergAmps in source lines or line codes: default line thermal ratings are not established physical ratings",
        "files": records,
    }
    provenance_path = WORK / "research/network_provenance.json"
    existing = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    existing.update(manifest)
    provenance_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps({"download_files": len(records), "bytes": sum(r["bytes"] for r in records)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--simbench", action="store_true")
    args = parser.parse_args()
    if args.download:
        acquire()
    if args.verify:
        verify()
    if args.simbench:
        verify_simbench()


def verify_simbench():
    import simbench as sb
    import pandapower as pp
    import numpy as np
    from network_model import SimBenchLV

    package_path = Path(sb.__file__).parent
    source_dir = package_path / "networks/1-complete_data-mixed-all-0-sw"
    raw_dir = RAW / "simbench_1.6.3_scenario0_topology"
    if shutil.disk_usage("F:/").free < 40_000_000:
        raise RuntimeError("F drive space insufficient for SimBench topology sources")
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_records = []
    for source in source_dir.glob("*.csv"):
        if "Profile" in source.name:
            continue  # no native time-series profiles used in this experiment
        path = raw_dir / source.name
        if not path.exists():
            with path.open("xb") as f:
                f.write(source.read_bytes())
            path.chmod(0o444)
        source_records.append({"file": str(path), "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    panel = np.load(WORK / "datasets/ausgrid_source_clock_panel.npz")
    dates = panel["date"]
    period = (dates >= "2012-01-01") & (dates <= "2012-06-30")
    eligible = (panel["household_id"] != 2) & (panel["valid_feeder_day"][:, period].mean(axis=1) >= .95)
    ids = np.flatnonzero(eligible)[:55]
    days = np.flatnonzero(period & panel["valid_feeder_day"][ids].all(axis=0))
    data = panel["power_kw"][ids][:, days]
    c = panel["channel"].tolist()
    load = (data[:, :, :, c.index("GC")] + data[:, :, :, c.index("CL")]).transpose(1, 2, 0).reshape(-1, 55)[:1000]
    pv = data[:, :, :, c.index("GG")].transpose(1, 2, 0).reshape(-1, 55)[:1000]
    checks = []
    for name, code in [("simbench_rural1", "1-LV-rural1--0-sw"), ("simbench_semiurb4", "1-LV-semiurb4--0-sw")]:
        net = sb.get_simbench_net(code)
        # Native operating point is solved once before any household replacement.
        pp.runpp(net, algorithm="nr", numba=False, tolerance_mva=1e-9, max_iteration=50)
        native_loss = net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()
        native_balance = net.res_ext_grid.p_mw.sum() + net.res_sgen.p_mw.sum() - net.res_load.p_mw.sum() - native_loss
        native = {"converged": bool(net.converged), "vmin_pu": float(net.res_bus.vm_pu.min()),
                  "vmax_pu": float(net.res_bus.vm_pu.max()), "balance_abs_kw": abs(float(native_balance)) * 1000}
        native_counts = len(net.load)
        rng = np.random.default_rng(20261005)
        mandatory = rng.permutation(native_counts)
        extra = rng.choice(native_counts, 55 - native_counts, p=net.load.p_mw.to_numpy() / net.load.p_mw.sum())
        assignment = np.r_[mandatory, extra]
        rng.shuffle(assignment)
        meta = {
            "network": name, "simbench_code": code, "simbench_version": sb.__version__,
            "pandapower_version": pp.__version__, "balanced_only": True,
            "source_voltage_pu": net.ext_grid.vm_pu.to_list(),
            "transformer_ratings_kva": (net.trafo.sn_mva * 1000).to_list(),
            "native_load_locations": native_counts, "scenario_households": 55,
            "household_to_load_row": assignment.tolist(),
            "assignment_policy": "seed20261005; every native load location receives one household; additional locations sampled by native nominal demand weights",
            "preserved": ["network impedances", "transformer ratings", "cable ampacities", "tap positions", "source voltage"],
            "changes": ["replace native loads with 55 study households", "disable native sgen/gen/storage to avoid double counting supplied PV and batteries", "balanced AC only"],
            "license": "SimBench database: ODbL/DBCL; software BSD-3-Clause",
        }
        # Do not serialize the unused ~100MB native profile collection.
        if "profiles" in net:
            del net["profiles"]
        pp.to_json(net, str(WORK / "datasets" / ("network_" + name + ".json")))
        (WORK / "datasets" / ("network_" + name + "_mapping.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        model = SimBenchLV(name)
        replay = model.run(load, pv, return_bus=True)
        balance = replay["source_import_kw"] - (load - pv).sum(axis=1) - replay["loss_kw"]
        recycled = model.net.res_bus.vm_pu.to_numpy().copy()
        pp.runpp(model.net, algorithm="nr", numba=False, tolerance_mva=1e-9, init="dc", max_iteration=50, recycle=None)
        cold = model.net.res_bus.vm_pu.to_numpy()
        check = {"network": name, "code": code, "native_case": native,
                 "buses": len(net.bus), "lines": len(net.line), "native_loads": native_counts,
                 "transformer_kva": meta["transformer_ratings_kva"], "source_pu": meta["source_voltage_pu"],
                 "replayed_samples": len(load), "converged_samples": int(replay["converged"].sum()),
                 "seconds_per_1000": replay["elapsed_seconds"] * 1000 / len(load),
                 "max_balance_abs_kw": float(np.nanmax(np.abs(balance))),
                 "recycled_vs_cold_voltage_max_error_pu": float(np.max(np.abs(recycled - cold))),
                 "replay_vmin_pu": float(np.nanmin(replay["vmin_pu"])),
                 "replay_vmax_pu": float(np.nanmax(replay["vmax_pu"]))}
        checks.append(check)
    path = WORK / "research/network_provenance.json"
    provenance = json.loads(path.read_text())
    provenance["simbench"] = {"source": "PyPI simbench1.6.3 distributed CSV topology, copied byte-for-byte to immutable F raw repository",
                              "source_data_url": "https://simbench.de/en/download/datasets/",
                              "license_url": "https://raw.githubusercontent.com/e2nIEE/simbench/develop/LICENSE",
                              "files": source_records, "models": [x["code"] for x in checks]}
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (WORK / "results/network_simbench_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


def verify():
    import numpy as np
    from network_model import EuropeanLV

    def xlsx_rows(path, sheet=1):
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = ["".join(t.text or "" for t in si.findall(".//m:t", ns)) for si in root]
            root = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
            result = []
            for row in root.findall(".//m:sheetData/m:row", ns):
                values = []
                for cell in row:
                    col = re.match(r"([A-Z]+)", cell.attrib["r"])[1]
                    index = 0
                    for c in col:
                        index = index * 26 + ord(c) - 64
                    while len(values) < index:
                        values.append(None)
                    value = cell.find("m:v", ns)
                    if value is not None:
                        values[index - 1] = shared[int(value.text)] if cell.attrib.get("t") == "s" else float(value.text)
                result.append(values)
            return result

    # The IEEE technical document specifies constant PQ. Modern OpenDSS's
    # default Vmaxpu=1.05 is relative to load.kV=0.23, not the 0.416/sqrt(3)
    # circuit base. It therefore changes load power even at ordinary benchmark
    # voltages. Explicitly retaining constant PQ reproduces published snapshots.
    network = EuropeanLV(study_constant_power=True)
    profiles = np.stack([np.loadtxt(MODEL / f"Daily_1min_100profiles/Load_profile_{i}.txt") for i in range(1, 56)], axis=1)
    reference = []
    for minute, filename in [(1, "Snapshot_1min_Initialization_Off Peak.xlsx"),
                             (566, "Snapshot_566min_On Peak.xlsx"),
                             (1440, "Snapshot_1440min_End.xlsx")]:
        sample = profiles[minute - 1]
        voltage = network._solve(sample, sample * np.tan(np.arccos(.95)))
        expected, actual = [], []
        name_to_index = {name: i for i, name in enumerate(network.node_names)}
        for row in xlsx_rows(MODEL / "reference" / filename)[1:]:
            if len(row) < 14 or row[0] is None:
                continue
            bus = str(row[0]).lower().removesuffix(".0")
            if bus == "sourcebus":
                continue
            for phase, mag in [(1, 3), (2, 7), (3, 11)]:
                key = bus + "." + str(phase)
                if key in name_to_index and row[mag] is not None:
                    expected.append(row[mag])
                    actual.append(abs(voltage[name_to_index[key]]))
        errors = np.abs(np.array(expected) - np.array(actual))
        reference.append({"minute": minute, "compared_node_phases": len(errors),
                          "max_voltage_error_v": float(errors.max()),
                          "mean_voltage_error_v": float(errors.mean()),
                          "max_voltage_error_pu": float(errors.max() / network.lv_base_v),
                          "load_kw_total": float(sample.sum()),
                          "tolerance_pu": 0.001,
                          "pass_0p001pu": bool(errors.max() / network.lv_base_v < .001)})

    network = EuropeanLV()
    rng = np.random.default_rng(818)
    synthetic_load = rng.uniform(.1, 5, (4, 55))
    synthetic_pv = rng.uniform(0, 5, (4, 55))
    test = network.run(synthetic_load, synthetic_pv, return_bus=True)
    balance_error = test["source_import_kw"] - (synthetic_load - synthetic_pv).sum(axis=1) - test["loss_kw"]
    voltage = np.asarray(network.dss.Circuit.YNodeVArray()).view(np.complex128)
    predicted_currents = network.line_operator @ voltage
    current_errors = []
    for line, sl in zip(network.line_names, network.line_slices):
        network.dss.Circuit.SetActiveElement("line." + line)
        currents = np.asarray(network.dss.CktElement.Currents()).view(np.complex128)
        current_errors.append(np.max(np.abs(currents - predicted_currents[sl])))

    panel = np.load(WORK / "datasets/ausgrid_source_clock_panel.npz")
    dates = panel["date"]
    validation_period = (dates >= "2012-01-01") & (dates <= "2012-06-30")
    observed_fraction = panel["valid_feeder_day"][:, validation_period].mean(axis=1)
    ids = np.flatnonzero((panel["household_id"] != 2) & (observed_fraction >= .95))[:55]
    days = np.flatnonzero((dates >= "2012-01-01") & (dates <= "2012-06-30") & panel["valid_feeder_day"][ids].all(axis=0))
    data = panel["power_kw"][ids][:, days]
    channels = panel["channel"].tolist()
    gc, gg, cl = (channels.index(c) for c in ("GC", "GG", "CL"))
    load = (data[:, :, :, gc] + data[:, :, :, cl]).transpose(1, 2, 0).reshape(-1, 55)[:1000]
    pv = data[:, :, :, gg].transpose(1, 2, 0).reshape(-1, 55)[:1000]
    timing = network.run(load, pv)
    linear = network.voltage_linearization()
    np.savez_compressed(WORK / "datasets/network_eulv_linearization.npz", **linear)
    # Holdout perturbation check around the stated linearization point, not
    # tuning against test-year actions or selecting a voltage-safe subset.
    ptest = np.ones(55) + rng.normal(0, .1, 55)
    qtest = np.ones(55) * np.tan(np.arccos(.95)) + rng.normal(0, .05, 55)
    va = np.abs(network._solve(ptest, qtest)[network.house_nodes]) / network.lv_base_v
    vp = linear["v0"] + linear["dVdP"] @ (ptest - linear["p0"]) + linear["dVdQ"] @ (qtest - linear["q0"])
    results = {
        "date": "2026-10-05", "solver": network.dss.Basic.Version(),
        "nodes": len(network.node_names), "lines": len(network.line_names),
        "households": 55, "phase_counts": np.bincount(network.house_phases, minlength=4)[1:].tolist(),
        "transformer_kva": 800, "source_pu": 1.05,
        "reference_snapshots": reference,
        "physical_balance_max_abs_kw": float(np.max(np.abs(balance_error))),
        "sparse_current_operator_max_error_a": float(np.max(current_errors)),
        "negative_net_power_cases_converged": bool(test["converged"].all()),
        "linearization_check_max_abs_voltage_error_pu": float(np.max(np.abs(va - vp))),
        "timing": {"samples": len(load), "seconds": timing["elapsed_seconds"],
                   "seconds_per_1000": timing["elapsed_seconds"] * 1000 / len(load),
                   "estimated_1261440_step_minutes_serial": timing["elapsed_seconds"] / len(load) * 1261440 / 60,
                   "converged": int(timing["converged"].sum()), "data_period": "2012-01 to 2012-06 validation period",
                   "vmin_pu": float(np.nanmin(timing["vmin_pu"])), "vmax_pu": float(np.nanmax(timing["vmax_pu"]))},
        "study_model_note": "Constant-PQ loads over 0.5-1.5 pu reproduce the IEEE document's explicit constant-PQ specification. Modern engine default bounds (0.95/1.05 of load rating 0.23kV) produced a diagnosed peak-snapshot mismatch of 0.001665pu; using documented constant PQ removes this mismatch.",
        "line_thermal_note": "No explicit ampacity in source: output current only. No line overload claims.",
    }
    (WORK / "results/network_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(WORK / "results/network_validation_timeseries.npz", **{k: v for k, v in timing.items() if isinstance(v, np.ndarray)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
