"""Separate retail settlement, throughput proxy and operating cost on paired data."""
from pathlib import Path
import json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main(args):
    splits=json.loads((ROOT/'datasets/splits_v1.json').read_text())
    dest=ROOT/'results/analysis';dest.mkdir(exist_ok=True)
    records=[]
    for file in sorted((ROOT/'results/dispatch').glob(f'*_test_{args.suffix}.npz')):
        status=file.with_suffix('.json')
        if not status.exists():continue
        st=json.loads(status.read_text());model=st.get('model',st.get('method'))
        if st.get('n_failed',st.get('failed',0)):raise ValueError(file)
        z=np.load(file);ids=z['household_id'];wear=.5*.02*(z['charge_kw']+z['discharge_kw']).sum(2)
        retail=z['bill_aud']-wear
        oracle=np.load(ROOT/f'results/dispatch/oracle_{st["tariff"]}_seed11_test_{args.suffix}.npz')
        if not np.array_equal(ids,oracle['household_id']):raise ValueError('Household order mismatch')
        for panel,group in enumerate(splits['primary_panels']+[splits['heldout_network_household_ids']]):
            rows=np.array([list(ids).index(h) for h in group]);valid=z['clean_day'][rows].all(0)
            if not np.array_equal(valid,oracle['clean_day'][rows].all(0)):raise ValueError('Oracle mask differs')
            expected=splits['primary_panel_common_clean_dates'][panel]['test'] if panel<3 else splits['heldout_network_common_clean_dates']['test']
            if not np.array_equal(valid,np.isin(z['date'],expected)):raise ValueError('Frozen mask differs')
            bill=z['bill_aud'][rows][:,valid];regret=bill-oracle['bill_aud'][rows][:,valid]
            learned=model not in ['none','self']
            if learned and np.nanmin(regret)<-1e-5:raise ValueError(f'Below perfect forecast cost {model} {regret.min()}')
            rec={'model':model,'tariff':st['tariff'],'seed':st['seed'],'panel':panel,'n_households':55,'n_clean_dates':int(valid.sum()),
              'operating_cost_aud_per_household_day':float(bill.mean()),
              'retail_electricity_bill_aud_per_household_day':float(retail[rows][:,valid].mean()),
              'throughput_proxy_aud_per_household_day':float(wear[rows][:,valid].mean()),
              'energy_throughput_kwh_per_household_day':float(wear[rows][:,valid].mean()/.02),
              'perfect_forecast_regret_aud_per_household_day':float(regret.mean()) if learned else None}
            if 'forecast_pv_kw' in z.files:
                err=z['forecast_pv_kw'][rows][:,valid]-z['actual_pv_kw'][rows][:,valid]
                rec['pv_rmse_kw']=float(np.sqrt(np.mean(err**2)));rec['pv_bias_kw']=float(err.mean())
                rec['forecast_night_energy_kwh_per_day']=float(.5*np.concatenate([z['forecast_pv_kw'][rows][:,valid,:8],z['forecast_pv_kw'][rows][:,valid,42:]],axis=2).sum(2).mean())
            if model=='self':rec['final_minus_initial_kwh_mean']=float(np.mean(z['end_energy_kwh'][rows,-1]-5))
            records.append(rec)
    out={'status':'descriptive paired coverage; no model selection','cost_definition':'retail settlement including controlled load plus 0.02 AUD/kWh throughput proxy; no CAPEX or service fee','records':records}
    out['regime']=args.suffix
    filename='dispatch_cost_components.json' if args.suffix=='tight' else f'dispatch_cost_components_{args.suffix}.json'
    (dest/filename).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'records':len(records),'models':sorted(set(str(x['model']) for x in records))}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suffix',default='tight',choices=['tight','physical']);main(ap.parse_args())
