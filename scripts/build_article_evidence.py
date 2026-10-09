"""Build compact, traceable display values; never fit or select a model."""
from pathlib import Path
import json
import numpy as np
from analyze_network import metrics
from analyze_robustness import paired
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'results/analysis'
def read(name):return json.loads((A/name).read_text(encoding='utf-8'))
def main():
    out={'source_root':'results/analysis','main':[],'heldout':[],'competitors':[],'robustness':[],'controls':[]}
    for group,tag in [('main','main_physical'),('heldout','heldout_physical')]:
        f=A/f'{tag}_vs_msecont_paired.json'
        if not f.exists():continue
        for r in read(f.name)['records']:
            if r['battery_count']==28 and r['tariff_mix']!='all_tou':continue
            row={k:r[k] for k in ['panel','battery_count','tariff_mix']}
            row['source']=f.name
            row['cost_owner_cents']=dict(r['operating_cost_delta_per_battery_owner_aud'])
            row['cost_owner_cents']['mean']*=100
            row['cost_owner_cents']['ci95']=[v*100 for v in row['cost_owner_cents']['ci95']]
            for short,key in [('voltage_pp','voltage_percentage_points'),('peak_kw','daily_import_peak_kw'),('export_kw','daily_export_peak_kw'),('loss_kwh','loss_kwh')]:
                row[short]=r['outcomes'][key]
            out[group].append(row)
    f=A/'dispatch_cost_components_physical.json'
    if f.exists():
        rows=read(f.name)['records']
        for panel in [0,3]:
            for m in ['none','persistence','hgb','mlp','msecont','calibrated','dfl','stochastic','oracle','self']:
                rr=[r for r in rows if r['panel']==panel and r['model']==m and r['tariff']=='tou']
                if not rr:continue
                record={'panel':panel,'model':m,'seeds':[r['seed'] for r in rr],'source':f.name}
                for k in ['operating_cost_aud_per_household_day','retail_electricity_bill_aud_per_household_day','energy_throughput_kwh_per_household_day','pv_rmse_kw']:
                    vv=[r[k] for r in rr if k in r]
                    record[k]=float(np.mean(vv)) if vv else None
                out['competitors'].append(record)
    f=A/'robustness_pairs_physical.json'
    if f.exists():
        r=read(f.name);out['robustness']=r['external_mapping_pairs'];out['controls']=r['controls']
    for key,file in [('decomposition','mechanism_attribution_physical.json'),('oracle_diagnostic','signal_semantics_physical.json')]:
        if (A/file).exists():out[key]=read(file)
    out['strong_network']=[]
    for folder in ['main_physical','strong_baselines_physical']:
        for file in sorted((ROOT/'results/network_replay'/folder).glob('*.json')):
            st=json.loads(file.read_text())
            if st.get('status')!='completed' or st.get('panel')!=0 or st.get('battery_count')!=55 or st.get('seed')!=11 or st.get('tariff_mix')!='all_tou':continue
            with np.load(file.with_suffix('.npz')) as z:vals={k:float(np.nanmean(v)) for k,v in metrics(z).items()}
            out['strong_network'].append({'model':st['objective'],'source':str(file.relative_to(ROOT)),'outcomes':vals})
    out['alternative_minus_dfl']={}
    found={r['model']:ROOT/r['source'] for r in out['strong_network']}
    if 'dfl' in found:
        for m in ['stochastic','calibrated','hgb']:
            if m not in found:continue
            with np.load(found['dfl'].with_suffix('.npz')) as a,np.load(found[m].with_suffix('.npz')) as b:out['alternative_minus_dfl'][m]=paired(a,b)
    target=ROOT/'manuscript/article_evidence.json';target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:len(v) for k,v in out.items() if isinstance(v,list)}))
    for r in out['main']:
        print('MAIN',r['panel'],r['battery_count'],r['tariff_mix'],'cents',r['cost_owner_cents'],'voltage',r['voltage_pp']['delta_dfl_minus_mse'])
    for r in out['competitors']:print('MODEL',r)
if __name__=='__main__':main()
