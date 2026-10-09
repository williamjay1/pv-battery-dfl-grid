"""Small LP feasibility check, using synthetic data only. NOT a DFL experiment.
Run in a Python environment with NumPy/SciPy. --deps may point at a task-local installation.
The LP omits the unique quadratic tie-break required by the final protocol.
"""
import argparse, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--deps')
ap.add_argument('--out', required=True)
ap.add_argument('--csv', help='Optional genuine Ausgrid annual CSV or partial preview, with one preamble row.')
args = ap.parse_args()
if args.deps:
    sys.path.insert(0, args.deps)
import numpy as np
from scipy.optimize import linprog

def solve(net, buy, sell=0.08, dt=0.5, cap=10.0, power=3.0, eta=0.95):
    n=len(net)
    # Variables: charge, discharge, import, export, and energy after each slot.
    c=np.r_[np.full(n,0.02*dt),np.full(n,0.02*dt),buy*dt,
            np.full(n,-sell*dt),np.zeros(n)]
    A=np.zeros((2*n+1,5*n)); b=np.zeros(2*n+1)
    for t in range(n):
        A[t,t]=-1; A[t,n+t]=1; A[t,2*n+t]=1; A[t,3*n+t]=-1
        b[t]=net[t]
        A[n+t,t]=-eta*dt; A[n+t,n+t]=dt/eta; A[n+t,4*n+t]=1
        if t: A[n+t,4*n+t-1]=-1
        else: b[n+t]=cap*0.5
    A[-1,-1]=1; b[-1]=cap*0.5
    bounds=[(0,power)]*(2*n)+[(0,None)]*(2*n)+[(0.1*cap,0.9*cap)]*n
    res=linprog(c,A_eq=A,b_eq=b,bounds=bounds,method='highs')
    if not res.success: raise RuntimeError(res.message)
    x=res.x
    assert np.max(np.abs(A@x-b))<1e-6
    assert np.min(x[4*n:])>=cap*0.1-1e-6
    assert np.max(x[4*n:])<=cap*0.9+1e-6
    assert np.max(np.minimum(x[:n],x[n:2*n]))<1e-6
    return {"objective_aud":float(res.fun),"max_balance_residual":float(np.max(np.abs(A@x-b))),
            "terminal_energy_kwh":float(x[-1]),"max_simultaneous_charge_discharge_kw":float(np.max(np.minimum(x[:n],x[n:2*n])))}

t=np.arange(48)/2
pv=np.maximum(0,3*np.sin(np.pi*(t-6)/12))
load=0.5+1.5*np.exp(-((t-19)/2)**2)
input_info='synthetic 48-slot example; no empirical finding'
if args.csv:
    import csv
    with open(args.csv,encoding='utf-8-sig',newline='') as f:
        next(f)
        reader=csv.DictReader(f)
        slots=[f'{k//2}:{30 if k%2 else 0:02d}' for k in range(1,48)]+['0:00']
        rows={}
        for row in reader:
            key=(row['Customer'],row['date'])
            rows.setdefault(key,{})[row['Consumption Category']]=row
            if 'GC' in rows[key] and 'GG' in rows[key]:
                load=np.array([float(rows[key]['GC'][s])*2 for s in slots])
                pv=np.array([float(rows[key]['GG'][s])*2 for s in slots])
                input_info=f'real downloaded Ausgrid preview: customer={key[0]}, date={key[1]}; GC and GG only; kWh/0.5h converted to kW'
                break
        else: raise ValueError('No complete GC/GG household-day in supplied CSV')
buy=np.where((t<7)|(t>=22),0.18,np.where((t>=16)&(t<21),0.50,0.30))
start=time.perf_counter()
r=solve(load-pv,buy)
r.update({"status":"PASS","input":input_info,
          "scope":"LP energy balance and feasible battery trajectory only; no DFL, AC power flow, or differentiable QP verified",
          "wall_seconds":time.perf_counter()-start})
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False))
