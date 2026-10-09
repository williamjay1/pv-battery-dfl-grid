"""Unbalanced AC replay of the public EPRI European LV benchmark.

The 55 inputs are household kW; battery kW is positive when charging. Reactive
power is calculated only from gross household demand. PV and batteries have
unity power factor. Original impedances, phases, transformer rating, and source
voltage are retained. Lines have no published thermal ratings in this source;
line currents are returned, but no physical line-overload claim is supported.
"""
from __future__ import annotations

from pathlib import Path
import re
import time
import numpy as np

WORK = Path(__file__).resolve().parents[1]
MODEL = WORK / "datasets/network_eulv"


class EuropeanLV:
    n_households = 55
    transformer_kva = 800.0

    def __init__(self, model_dir=MODEL, source_pu=1.05, study_constant_power=True):
        import opendssdirect as odd
        from scipy.sparse import csr_matrix

        self.dss = odd.NewContext()
        dss = self.dss
        dss.Basic.AllowChangeDir(False)
        dss.Basic.AllowEditor(False)
        dss.Basic.AllowForms(False)
        model_dir = Path(model_dir)
        dss.Text.Command("clear")
        dss.Text.Command(f'set datapath="{WORK / "results"}"')
        dss.Text.Command("set defaultbasefrequency=50")
        dss.Text.Command("new circuit.LVTest")
        dss.Text.Command(f"edit vsource.source basekv=11 pu={source_pu} isc3=3000 isc1=5")
        for name in ("LineCode.txt", "Lines.txt", "Transformers.txt"):
            dss.Text.Command(f'redirect "{model_dir / name}"')
        # Avoid source loadshape time progression. Actual samples are supplied
        # explicitly by the caller and are never normalized by a profile peak.
        for line in (model_dir / "Loads.txt").read_text().splitlines():
            line = re.sub(r"\s+Yearly=\S+", "", line, flags=re.I).strip()
            if line:
                dss.Text.Command(line)
        if study_constant_power:
            dss.Text.Command("batchedit load..* model=1 vminpu=0.5 vmaxpu=1.5")
        dss.Text.Command("set voltagebases=[11 .416]")
        dss.Text.Command("calcvoltagebases")
        dss.Text.Command("set mode=snapshot controlmode=off maxiterations=100 tolerance=0.0000001")
        dss.Solution.Solve()
        self.load_names = list(dss.Loads.AllNames())
        if len(self.load_names) != 55:
            raise ValueError(f"Expected 55 benchmark households, found {len(self.load_names)}")
        self.node_names = [n.lower() for n in dss.Circuit.YNodeOrder()]
        lookup = {n: i for i, n in enumerate(self.node_names)}
        self.house_nodes, self.house_phases, self.house_buses = [], [], []
        for name in self.load_names:
            dss.Circuit.SetActiveElement("load." + name)
            bus = dss.CktElement.BusNames()[0].lower()
            self.house_nodes.append(lookup[bus])
            self.house_phases.append(int(bus.split(".")[1]))
            self.house_buses.append(bus.split(".")[0])
        self.house_nodes = np.array(self.house_nodes, dtype=int)
        self.house_phases = np.array(self.house_phases, dtype=int)
        self.lv_base_v = 416 / np.sqrt(3)

        # Sparse line terminal-current operator. This avoids 905 Python/C API
        # round trips per step and is checked against direct element currents.
        rows, cols, data = [], [], []
        self.line_names = list(dss.Lines.AllNames())
        self.line_slices = []
        row0 = 0
        for line in self.line_names:
            dss.Circuit.SetActiveElement("line." + line)
            ncond = dss.CktElement.NumConductors()
            nterm = dss.CktElement.NumTerminals()
            n = ncond * nterm
            yp = np.asarray(dss.CktElement.YPrim(), dtype=float).view(np.complex128).reshape((n, n), order="F")
            order = dss.CktElement.NodeOrder()
            buses = dss.CktElement.BusNames()
            node_ids = []
            for terminal in range(nterm):
                root = buses[terminal].split(".")[0].lower()
                for conductor in range(ncond):
                    phase = order[terminal * ncond + conductor]
                    node_ids.append(lookup.get(root + "." + str(phase), -1))
            # Both terminals are retained for diagnostic parity (C=0 here).
            for r in range(n):
                for c, global_node in enumerate(node_ids):
                    if global_node >= 0 and abs(yp[r, c]) > 0:
                        rows.append(row0 + r)
                        cols.append(global_node)
                        data.append(yp[r, c])
            self.line_slices.append(slice(row0, row0 + n))
            row0 += n
        self.line_operator = csr_matrix((data, (rows, cols)), shape=(row0, len(self.node_names)))
        self.source_pu = source_pu

    def _solve(self, net_kw, reactive_kvar):
        dss = self.dss
        for i in range(55):
            dss.Loads.Idx(i + 1)
            dss.Loads.kW(float(net_kw[i]))
            # Explicit Q after kW avoids an unintended signed power-factor rule
            # when the aggregate net active demand is negative.
            dss.Loads.kvar(float(reactive_kvar[i]))
        dss.Solution.Solve()
        voltage = np.asarray(dss.Circuit.YNodeVArray(), dtype=float).view(np.complex128)
        return voltage

    def run(self, load_kw, pv_kw, battery_kw=None, *, mapping=None,
            power_factor=0.95, voltage_band=(0.95, 1.05), return_bus=False):
        load = np.atleast_2d(np.asarray(load_kw, dtype=float))
        pv = np.atleast_2d(np.asarray(pv_kw, dtype=float))
        battery = np.zeros_like(load) if battery_kw is None else np.atleast_2d(np.asarray(battery_kw, dtype=float))
        if load.shape != pv.shape or load.shape != battery.shape or load.shape[1] != 55:
            raise ValueError("load, pv, battery must have the same [time,55] shape")
        if not (0 < power_factor <= 1):
            raise ValueError("power factor must be in (0,1]")
        if mapping is not None:
            mapping = np.asarray(mapping, dtype=int)
            if len(mapping) != 55 or sorted(mapping.tolist()) != list(range(55)):
                raise ValueError("mapping must be a permutation: household index for each benchmark load")
            load, pv, battery = (a[:, mapping] for a in (load, pv, battery))
        if np.any(load < -1e-10) or np.any(pv < -1e-10):
            raise ValueError("gross load and PV must be nonnegative")
        n = len(load)
        columns = ["vmin_pu", "vmax_pu", "voltage_violation_fraction", "voltage_exceedance_pu",
                   "transformer_loading_pu", "transformer_max_phase_loading_pu",
                   "line_max_current_a", "loss_kw", "source_import_kw", "source_reactive_kvar"]
        out = {name: np.full(n, np.nan) for name in columns}
        out["converged"] = np.zeros(n, dtype=bool)
        out["input_valid"] = np.isfinite(load + pv + battery).all(axis=1)
        if return_bus:
            out["household_voltage_pu"] = np.full((n, 55), np.nan)
        qmult = np.tan(np.arccos(power_factor))
        start = time.perf_counter()
        for t in range(n):
            if not out["input_valid"][t]:
                continue
            v = self._solve(load[t] - pv[t] + battery[t], load[t] * qmult)
            if not self.dss.Solution.Converged():
                continue
            out["converged"][t] = True
            vh = np.abs(v[self.house_nodes]) / self.lv_base_v
            out["vmin_pu"][t] = vh.min()
            out["vmax_pu"][t] = vh.max()
            out["voltage_violation_fraction"][t] = np.mean((vh < voltage_band[0]) | (vh > voltage_band[1]))
            out["voltage_exceedance_pu"][t] = np.maximum(voltage_band[0] - vh, 0).sum() + np.maximum(vh - voltage_band[1], 0).sum()
            out["line_max_current_a"][t] = np.abs(self.line_operator @ v).max()
            out["loss_kw"][t] = self.dss.Circuit.Losses()[0] / 1000
            out["source_import_kw"][t], out["source_reactive_kvar"][t] = -np.array(self.dss.Circuit.TotalPower())
            self.dss.Circuit.SetActiveElement("transformer.tr1")
            powers = np.asarray(self.dss.CktElement.Powers()).reshape(2, 4, 2)
            # LV winding phase powers; neutral power is excluded.
            phase_pq = powers[1, :3]
            apparent = np.linalg.norm(phase_pq.sum(axis=0))
            out["transformer_loading_pu"][t] = apparent / self.transformer_kva
            out["transformer_max_phase_loading_pu"][t] = np.linalg.norm(phase_pq, axis=1).max() / (self.transformer_kva / 3)
            if return_bus:
                out["household_voltage_pu"][t] = vh
        out["elapsed_seconds"] = time.perf_counter() - start
        out["metadata"] = {
            "network": "EPRI European LV test feeder", "transformer_kva": 800,
            "source_pu": self.source_pu, "load_power_factor": power_factor,
            "voltage_band": list(voltage_band), "band_is_study_criterion_not_legal_standard": True,
            "household_order": self.load_names, "mapping": None if mapping is None else mapping.tolist(),
            "line_rating_status": "unspecified; currents only, no physical loading percentage",
            "power_sign": "source import positive; household battery charging positive",
        }
        return out

    def voltage_linearization(self, load_kw=None, pv_kw=None, power_factor=0.95, delta_kw=0.1):
        """AC finite-difference sensitivity for a forecast-side QP constraint.

        Columns correspond to benchmark loads. Return v0, p0, q0, dV/dP and
        dV/dQ. P/Q are demand-positive kW/kvar. Linear constraints are a planning
        approximation; they do not certify AC feasibility. The caller must use
        forecast information only and keep full-day battery energy constraints.
        """
        load = np.ones(55) if load_kw is None else np.asarray(load_kw, dtype=float)
        pv = np.zeros(55) if pv_kw is None else np.asarray(pv_kw, dtype=float)
        p0 = load - pv
        q0 = load * np.tan(np.arccos(power_factor))
        v0 = np.abs(self._solve(p0, q0)[self.house_nodes]) / self.lv_base_v
        hp, hq = np.empty((55, 55)), np.empty((55, 55))
        for h in range(55):
            shift = np.zeros(55)
            shift[h] = delta_kw
            hp[:, h] = (np.abs(self._solve(p0 + shift, q0)[self.house_nodes]) / self.lv_base_v - v0) / delta_kw
            hq[:, h] = (np.abs(self._solve(p0, q0 + shift)[self.house_nodes]) / self.lv_base_v - v0) / delta_kw
        return {"v0": v0, "p0": p0, "q0": q0, "dVdP": hp, "dVdQ": hq,
                "house_phases": self.house_phases.copy()}


def simulate_network(load_kw, pv_kw, battery_kw=None, *, mapping=None,
                     network="eulv", power_factor=0.95, return_bus=False, source_pu=None, **kwargs):
    if network != "eulv" and source_pu is not None:
        raise ValueError("source_pu sensitivity is implemented only for eulv")
    model = EuropeanLV(source_pu=1.05 if source_pu is None else source_pu) if network == "eulv" else SimBenchLV(network)
    return model.run(load_kw, pv_kw, battery_kw, mapping=mapping,
                     power_factor=power_factor, return_bus=return_bus, **kwargs)


class SimBenchLV:
    """Balanced, supplementary topology test; not unbalanced confirmation.

    The same 55 household traces are assigned deterministically to the existing
    LV load locations. All 13 or 41 locations receive at least one household;
    remaining assignments use native nominal-demand weights. Original native
    PV/storage is disabled to avoid double-counting the supplied PV/batteries.
    Original transformer, cable ratings, tap positions, and source setpoint stay.
    """
    n_households = 55

    def __init__(self, network="simbench_rural1"):
        import pandapower as pp
        import json
        if network not in ("simbench_rural1", "simbench_semiurb4"):
            raise ValueError("network must be eulv, simbench_rural1 or simbench_semiurb4")
        self.pp = pp
        self.network = network
        self.net = pp.from_json(str(WORK / "datasets" / ("network_" + network + ".json")))
        meta = json.loads((WORK / "datasets" / ("network_" + network + "_mapping.json")).read_text())
        self.metadata = meta
        self.load_assignment = np.asarray(meta["household_to_load_row"], dtype=int)
        self.load_buses = self.net.load.bus.to_numpy(dtype=int)
        self.assignment = np.zeros((55, len(self.net.load)))
        self.assignment[np.arange(55), self.load_assignment] = 1
        self.net.load.loc[:, "scaling"] = 1.0
        self.net.load.loc[:, "const_z_p_percent"] = 0.0
        self.net.load.loc[:, "const_i_p_percent"] = 0.0
        self.net.load.loc[:, "const_z_q_percent"] = 0.0
        self.net.load.loc[:, "const_i_q_percent"] = 0.0
        for field in ("sgen", "gen", "storage"):
            if len(self.net[field]):
                self.net[field].loc[:, "in_service"] = False
        self.pp.runpp(self.net, algorithm="nr", numba=False, tolerance_mva=1e-9, max_iteration=50)

    def run(self, load_kw, pv_kw, battery_kw=None, *, mapping=None,
            power_factor=.95, voltage_band=(.95, 1.05), return_bus=False):
        load = np.atleast_2d(np.asarray(load_kw, dtype=float))
        pv = np.atleast_2d(np.asarray(pv_kw, dtype=float))
        battery = np.zeros_like(load) if battery_kw is None else np.atleast_2d(np.asarray(battery_kw, dtype=float))
        if load.shape != pv.shape or load.shape != battery.shape or load.shape[1] != 55:
            raise ValueError("load, pv, battery must have the same [time,55] shape")
        if np.any(load < -1e-10) or np.any(pv < -1e-10):
            raise ValueError("gross load and PV must be nonnegative")
        if mapping is not None:
            mapping = np.asarray(mapping, dtype=int)
            if len(mapping) != 55 or sorted(mapping.tolist()) != list(range(55)):
                raise ValueError("mapping must be a permutation")
            load, pv, battery = (a[:, mapping] for a in (load, pv, battery))
        if not (0 < power_factor <= 1):
            raise ValueError("power factor must be in (0,1]")
        n = len(load)
        fields = ["vmin_pu", "vmax_pu", "voltage_violation_fraction", "voltage_exceedance_pu",
                  "transformer_loading_pu", "line_loading_pu", "line_max_current_a", "loss_kw",
                  "source_import_kw", "source_reactive_kvar"]
        result = {k: np.full(n, np.nan) for k in fields}
        result["converged"] = np.zeros(n, bool)
        result["input_valid"] = np.isfinite(load + pv + battery).all(axis=1)
        if return_bus:
            result["household_voltage_pu"] = np.full((n, 55), np.nan)
        p = (load - pv + battery) @ self.assignment / 1000
        q = load @ self.assignment * np.tan(np.arccos(power_factor)) / 1000
        start = time.perf_counter()
        for t in range(n):
            if not result["input_valid"][t]:
                continue
            self.net.load.loc[:, "p_mw"] = p[t]
            self.net.load.loc[:, "q_mvar"] = q[t]
            try:
                self.pp.runpp(self.net, algorithm="nr", numba=False, init="results", tolerance_mva=1e-9, max_iteration=50,
                              recycle={"bus_pq": True, "gen": False, "trafo": False})
            except self.pp.LoadflowNotConverged:
                continue
            result["converged"][t] = True
            vv = self.net.res_bus.vm_pu.loc[self.load_buses].to_numpy()[self.load_assignment]
            result["vmin_pu"][t] = vv.min()
            result["vmax_pu"][t] = vv.max()
            result["voltage_violation_fraction"][t] = np.mean((vv < voltage_band[0]) | (vv > voltage_band[1]))
            result["voltage_exceedance_pu"][t] = np.maximum(voltage_band[0] - vv, 0).sum() + np.maximum(vv - voltage_band[1], 0).sum()
            result["transformer_loading_pu"][t] = self.net.res_trafo.loading_percent.max() / 100
            result["line_loading_pu"][t] = self.net.res_line.loading_percent.max() / 100
            result["line_max_current_a"][t] = self.net.res_line.i_ka.max() * 1000
            result["loss_kw"][t] = (self.net.res_line.pl_mw.sum() + self.net.res_trafo.pl_mw.sum()) * 1000
            result["source_import_kw"][t] = self.net.res_ext_grid.p_mw.sum() * 1000
            result["source_reactive_kvar"][t] = self.net.res_ext_grid.q_mvar.sum() * 1000
            if return_bus:
                result["household_voltage_pu"][t] = vv
        result["elapsed_seconds"] = time.perf_counter() - start
        result["metadata"] = {**self.metadata, "load_power_factor": power_factor,
                              "balanced_only": True, "voltage_band": list(voltage_band)}
        return result
