"""Independent output-to-equation checks, suitable for the local reproduction bundle."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify_expanded(root=ROOT):
    rev=root/'revision_20261009';files=sorted((rev/'dispatch').glob('*_seed11_expanded.npz'))
    raw=np.load(root/'datasets'/'ausgrid_source_clock_panel.npz')
    byid={int(x):i for i,x in enumerate(raw['household_id'])};bydate={str(x):i for i,x in enumerate(raw['date'])}
    report={'scope':'Independent settlement/state recomputation from measured panel and stored actions; not independent field validation.',
            'checks':[],'failures':[],'not_ready':[]}
    for path in files:
        if not path.with_suffix('.json').exists():
            report['not_ready'].append(str(path.relative_to(root)).replace('\\','/'))
            continue
        z=np.load(path); hs=np.array([byid[int(x)] for x in z['household_id']]); ds=np.array([bydate[str(x)] for x in z['date']])
        truth=raw['power_kw'][hs][:,ds].astype(float)
        gc=truth[:,:,:,0];pv=truth[:,:,:,2];cl=truth[:,:,:,1]
        c=z['charge_kw'].astype(float);d=z['discharge_kw'].astype(float); g=gc-pv+c-d
        tariff='flat' if '_flat_' in path.name else 'tou'; buy=np.full(48,.30)
        if tariff=='tou':buy[:14]=.18;buy[44:]=.18;buy[32:42]=.50
        retail=.5*np.sum(buy*np.maximum(g,0)+.08*np.minimum(g,0),axis=2)
        throughput=.5*np.sum(c+d,axis=2)
        operative=retail+.02*throughput
        full=operative+.09*np.sum(cl,axis=2)
        mask=z['valid_gcgg_day'].astype(bool); fullmask=z['clean_day'].astype(bool)
        errors={'excluding_cl':float(np.max(np.abs(operative[mask]-z['operating_excluding_cl_aud'][mask]))),
          'full_cost':float(np.max(np.abs(full[fullmask]-z['bill_aud'][fullmask]))),
          'retail':float(np.max(np.abs((retail+.09*np.sum(cl,axis=2))[fullmask]-z['retail_bill_aud'][fullmask]))),
          'throughput':float(np.max(np.abs(throughput[mask]-z['energy_throughput_kwh'][mask]))),
          'terminal_energy':float(np.max(np.abs((5+.5*np.sum(.95*c-d/.95,axis=2))[mask]-5)))}
        r={'file':str(path.relative_to(root)).replace('\\','/'),'sha256':sha(path),'gcgg_days':int(mask.sum()),'fullcost_days':int(fullmask.sum()),'max_abs_errors':errors,
          'mean_excluding_cl':float(operative[mask].mean()),'mean_full_cost':float(full[fullmask].mean()),
          'passed':max(errors.values())<1e-5}
        report['checks'].append(r)
        if not r['passed']: report['failures'].append(r['file'])
    reference=rev/'results'/'expanded_comparison_tou.json'
    if reference.exists():
        j=json.loads(reference.read_text());target=j['cohorts']['all299']['paired']['dfl_tou_seed11_expanded__minus__msecont_tou_seed11_expanded']['micro_mean_difference']
        a=np.load(rev/'dispatch'/'dfl_tou_seed11_expanded.npz');b=np.load(rev/'dispatch'/'msecont_tou_seed11_expanded.npz')
        assert np.array_equal(a['household_id'],b['household_id']) and np.array_equal(a['date'],b['date'])
        assert np.array_equal(a['valid_gcgg_day'],b['valid_gcgg_day'])
        m=a['valid_gcgg_day'];v=float((a['operating_excluding_cl_aud']-b['operating_excluding_cl_aud'])[m].mean())
        report['primary_delta_check']={'computed':v,'reported':target,'absolute_error':abs(v-target),'passed':abs(v-target)<1e-12}
    out=rev/'research'/'independent_settlement_verification.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'checks':len(report['checks']),'failures':report['failures'],'primary':report.get('primary_delta_check'),'report':str(out)}))
    if report['failures']:raise RuntimeError(report['failures'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);args=ap.parse_args();verify_expanded(args.root)
