"""Descriptive rating-exceedance summary from existing five-condition replay."""
import json
from pathlib import Path
import numpy as np
from revision_network_diagnostics import bootstrap
from revision_network_io import save_json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'revision_20261009'

def main():
 rows=[];pairs=[];data={}
 for r in json.loads((BASE/'network_diagnostics/scenarios.json').read_text())['records']:
  m=r['metadata'];p=m['parameters'] or {}
  if not(m['panel']==0 and m['seed']==11 and m['battery_count']==55 and m['tariff_mix']=='all_tou' and m['objective'] in ['msecont','dfl','mlp'] and p.get('mapping_seed') is None and p.get('power_factor',.95)==.95 and p.get('impedance_scale',1)==1 and p.get('export_limit_kw') is None):continue
  with np.load(ROOT/r['source']) as a:
   clean=a['clean_day'];loading=a['transformer_loading_pu'].reshape(-1,48)
   daily=np.where(clean,(loading>1).mean(axis=1)*100,np.nan);key=(p['network'],p['source_pu'])
   data[key+(m['objective'],)]=daily
   rows.append(dict(network=key[0],source_pu=key[1],method=m['objective'],observed_max_loading_pct=float(loading[clean].max()*100),fraction_clean_halfhours_above_rating_pct=float(np.nanmean(daily))))
 for network,source in sorted(set(k[:2] for k in data)):
  if (network,source,'msecont') in data and (network,source,'dfl') in data:
   pairs.append(dict(network=network,source_pu=source,dfl_minus_continued_transformer_above_rating_pp=bootstrap(data[(network,source,'dfl')]-data[(network,source,'msecont')])))
 save_json(BASE/'network_thermal_diagnostics.json',dict(interpretation='Descriptive transformer-rating exceedance at observed half-hourly snapshots, not transformer aging or dynamic thermal simulation. EULV apparent-power loading; SimBench native pandapower loading metric.',records=rows,paired=pairs))
 print(json.dumps(dict(status='completed',records=len(rows))))
if __name__=='__main__':main()
