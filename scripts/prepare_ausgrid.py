"""Create an auditable D-drive panel from the immutable downloaded Ausgrid ZIP.
No train/test fitting or interpolation is performed. Requires numpy and pandas.
Time is a source-clock date plus 48 interval indices, NOT a guessed UTC timestamp.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--raw-zip',required=True)
ap.add_argument('--out-dir',required=True)
args=ap.parse_args()
src=Path(args.raw_zip); out=Path(args.out_dir)
if out.drive.upper()!='D:':
    raise ValueError('Processed research outputs must be on D: under the current storage policy')
out.mkdir(parents=True,exist_ok=True)
target=out/'ausgrid_source_clock_panel.npz'
if target.exists(): raise FileExistsError('Use a new output version; existing panel will not be overwritten')
expected='5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82'
digest=hashlib.sha256(src.read_bytes()).hexdigest()
if digest!=expected: raise ValueError('ZIP differs from verified source version; inspect provenance before continuing')
dates=pd.date_range('2010-07-01','2013-06-30',freq='D')
slots=[f'{k//2}:{30 if k%2 else 0:02d}' for k in range(1,48)]+['0:00']
channels=['GC','CL','GG']; H=300; N=len(dates)
power=np.full((H,N,48,3),np.nan,dtype=np.float32)
quality_estimated=np.zeros((H,N,3),dtype=bool)
present=np.zeros((H,N,3),dtype=bool)
capacity=np.full((H,N),np.nan,dtype=np.float32)
audit=[]
with zipfile.ZipFile(src) as z:
    bad=z.testzip()
    if bad: raise ValueError(f'ZIP CRC failure: {bad}')
    members=sorted(n for n in z.namelist() if n.lower().endswith('.csv'))
    if len(members)!=3: raise ValueError(f'Expected exactly three annual CSVs, found {members}')
    for member in members:
        with z.open(member) as f:
            df=pd.read_csv(f,skiprows=1,keep_default_na=False)
        date_format='%d-%b-%y' if '2010-2011' in member else '%d/%m/%Y'
        d=pd.to_datetime(df['date'],format=date_format)
        hi=df['Customer'].to_numpy(dtype=int)-1
        di=(d-dates[0]).dt.days.to_numpy()
        if df.duplicated(['Customer','date','Consumption Category']).any(): raise ValueError('Duplicate row keys')
        counts={}
        for k,ch in enumerate(channels):
            mask=df['Consumption Category'].eq(ch).to_numpy()
            vals=df.loc[mask,slots].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float32)
            if np.any(~np.isfinite(vals)) or np.any(vals<0): raise ValueError('Invalid raw numeric value')
            power[hi[mask],di[mask],:,k]=vals*2.0
            present[hi[mask],di[mask],k]=True
            if 'Row Quality' in df:
                quality_estimated[hi[mask],di[mask],k]=df.loc[mask,'Row Quality'].eq('NA').to_numpy()
            if ch=='GG': capacity[hi[mask],di[mask]]=df.loc[mask,'Generator Capacity'].to_numpy(dtype=np.float32)
            counts[ch]=int(mask.sum())
        audit.append({'member':member,'rows':len(df),'channel_rows':counts})
never_cl=~present[:,:,1].any(axis=1)
power[never_cl,:,:,1]=0.0
present[never_cl,:,1]=True
valid_pair=present[:,:,0]&present[:,:,2]&~quality_estimated[:,:,0]&~quality_estimated[:,:,2]
valid_feeder=valid_pair&present[:,:,1]&~quality_estimated[:,:,1]
valid_pair[1,:]=False; valid_feeder[1,:]=False # predeclared incomplete customer 2
np.savez_compressed(target,power_kw=power,capacity_kwp=capacity,quality_estimated=quality_estimated,
    channel_present=present,valid_gc_gg_day=valid_pair,valid_feeder_day=valid_feeder,
    household_id=np.arange(1,301),date=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
    slot_end_label=np.array(slots),channel=np.array(channels))
manifest={'source_sha256':digest,'processed_panel':str(target),'shape':list(power.shape),
    'time_convention':'source date + interval index; final 0:00 is next midnight; DST unresolved',
    'power_unit':'kW, converted from raw half-hour kWh by multiplying by 2',
    'CL_policy':'never-observed CL treated as absent circuit zero; intermittent missing CL remains invalid',
    'excluded_customer_ids':[2],'estimated_quality_rows_excluded_from_masks':True,
    'not_done':['no imputation','no fit or split-based filtering','no final household panels selected','no DFL training','no AC simulation'],
    'valid_gc_gg_household_days':int(valid_pair.sum()),'valid_feeder_household_days':int(valid_feeder.sum()),
    'annual_file_audit':audit}
(out/'panel_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
