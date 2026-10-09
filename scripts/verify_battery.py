from pathlib import Path
import time,json
import numpy as np
import torch
from battery_model import Battery,DispatchQP,tariff,realized_bill
from decision_layer import BatteryLayer

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
rng=np.random.default_rng(1219);t=np.arange(48)*.5
net=.4+1.3*np.exp(-((t-19)/3)**2)-np.maximum(0,2.1*np.sin((t-6)*np.pi/12))+.04*rng.normal(size=48)
records=[]
for kind in ['tou','flat','equal']:
    qp=DispatchQP(kind=kind,tolerance=1e-12)
    c,d,e,diagnostics=qp.solve(net,True)
    buy,sell=tariff(kind)
    layer=BatteryLayer(kind=kind)
    x=torch.tensor(net,dtype=torch.float64,requires_grad=True)
    cc,dd,ee=layer(x)
    # Different net actual load keeps the true settlement away from training-forecast kinks.
    actual=net+.13*rng.normal(size=48)
    g=torch.tensor(actual)+cc-dd
    loss=.5*(torch.tensor(buy)*torch.relu(g)+torch.tensor(sell)*torch.minimum(g,torch.zeros_like(g))+.02*(cc+dd)).sum()
    start=time.perf_counter();loss.backward();elapsed=time.perf_counter()-start
    grad=x.grad.detach().numpy();directions=[]
    for j in range(5):
        direction=rng.normal(size=48);direction/=np.linalg.norm(direction)
        h=1e-4
        cp,dp,_=qp.solve(net+h*direction);cm,dm,_=qp.solve(net-h*direction)
        fd=(realized_bill(actual,cp,dp,buy,sell)-realized_bill(actual,cm,dm,buy,sell))/(2*h)
        ad=grad@direction
        directions.append({'finite_difference':float(fd),'adjoint':float(ad),'absolute_error':float(abs(fd-ad))})
    assert max(v['absolute_error'] for v in directions)<.003,(kind,directions)
    if kind=='equal':
        c2,d2,_=qp.solve(net+2*rng.normal(size=48))
        assert np.max(abs(c-c2))+np.max(abs(d-d2))<1e-7
    records.append({'tariff':kind,'diagnostics':diagnostics,'derivatives':directions,'backward_seconds':elapsed})
qp=DispatchQP();times=[]
for j in range(1000):
    start=time.perf_counter();qp.solve(net+.05*rng.normal(size=48));times.append(time.perf_counter()-start)
timing={'n':1000,'median_seconds':float(np.median(times)),'p95_seconds':float(np.quantile(times,.95))}
out={'status':'PASS','scope':'synthetic numerical audit; no empirical effect claim','checks':records,'qp_timing':timing,'derivative_backend':'Clarabel forward, rank-revealing active-set KKT adjoint, central differences. CVXPYlayers/diffcp unavailable on Windows; OSQP adjoint rejected after failed numerical gradient check.'}
(ROOT/'results/battery_numerical_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2),flush=True)
