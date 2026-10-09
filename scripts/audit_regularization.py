"""Independent HiGHS LP audit of the economic cost of unique QP selection."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
from battery_model import Battery,DispatchQP,realized_bill,tariff

ROOT=Path(__file__).resolve().parents[1]
def main():
    a=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    valid=a['valid_history']&a['valid_target_gcgg']&(a['split_code'][None,:]==1)&(a['household_role'][:,None]==0)
    pairs=np.argwhere(valid);rng=np.random.default_rng(8819);pairs=pairs[rng.choice(len(pairs),512,replace=False)]
    reports=[]
    for kind in ['tou','flat']:
        buy,sell=tariff(kind)
        qps={eps:DispatchQP(Battery(epsilon=eps),kind,tolerance=1e-13) for eps in [1e-5,1e-4,1e-3]}
        records=[]
        for h,d in pairs:
            net=a['gc_shared'][h,d]-a['pv_mlp_seed11'][h,d]
            row={}
            for eps,qp in qps.items():
                c,dd,e=qp.solve(net)
                if eps==1e-5:
                    lp=linprog(qp.q,A_ub=qp.A[49:],b_ub=qp.rhs[49:],A_eq=qp.A[:49],b_eq=qp.rhs[:49],bounds=[(None,None)]*qp.dim,method='highs')
                    if not lp.success:raise ValueError(lp.message)
                    economic_opt=realized_bill(net,lp.x[:48],lp.x[48:96],buy,sell)
                row[str(eps)]=float(realized_bill(net,c,dd,buy,sell)-economic_opt)
            records.append(row)
        reports.append({'tariff':kind,'n_validation_household_days':len(pairs),'gaps_aud':{str(eps):{'mean':float(np.mean([r[str(eps)] for r in records])),'max':float(np.max([r[str(eps)] for r in records])),'min':float(np.min([r[str(eps)] for r in records]))} for eps in qps}})
    out={'scope':'Independent original linear economic objective, excluding regularizer, on fixed validation sample; no test outcome selection','lp_solver':'scipy HiGHS','records':reports,'sample_pairs':pairs.tolist()}
    (ROOT/'results/regularization_economic_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
