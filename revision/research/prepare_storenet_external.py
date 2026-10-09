"""Prepare source-clock StoreNet cache under the prospectively frozen QC protocol.

No forecasting, fitting, dispatch, or outcome-based household selection occurs here.
"""
from pathlib import Path
import sys, json, hashlib, datetime
sys.path.insert(0, r'D:\MLWork\pv_battery_dfl_grid_20261005\temp\pydeps')
import numpy as np
import pandas as pd

ROOT = Path(r'D:\MLWork\pv_battery_dfl_grid_20261005\revision_20261009')
research = ROOT / 'research'
cache = ROOT / 'cache'
cache.mkdir(exist_ok=True)
manifest = json.loads((research / 'storenet_download_manifest.json').read_text())
protocol = research / 'storenet_external_protocol.json'
date = pd.date_range('2020-01-01', '2020-12-31', freq='D')
bins = pd.date_range('2020-01-01', '2020-12-31 23:30', freq='30min')
households, powers, valids, coverages, quality = [], [], [], [], []
for record in sorted(manifest['files'], key=lambda a: a['household']):
    if 'failed' in record:
        raise RuntimeError(record)
    path = Path(record['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    t = pd.to_datetime(data['date'], errors='coerce')
    gc = pd.to_numeric(data['Consumption(W)'], errors='coerce')
    pv = pd.to_numeric(data['Production(W)'], errors='coerce')
    duplicates = t.duplicated(keep=False)
    nonminute = t.notna() & (t != t.dt.floor('min'))
    finite = np.isfinite(gc) & np.isfinite(pv)
    negative = (gc < 0) | (pv < 0)
    eligible = finite & ~negative & ~duplicates & ~nonminute & t.notna()
    solar_count = int((eligible & (pv > 10)).sum())
    has_pv = solar_count >= 10
    q = dict(household_id=int(record['household']), raw_rows=len(data),
             raw_sha256=record['sha256'], raw_path=str(path),
             invalid_timestamp_rows=int(t.isna().sum()), duplicate_timestamp_rows=int(duplicates.sum()),
             nonminute_rows=int(nonminute.sum()), nonfinite_power_rows=int((~finite).sum()),
             negative_power_rows=int(negative.sum()), positive_pv_minutes_above_10W=solar_count,
             included_pv_household=has_pv, source_min=str(t.min()), source_max=str(t.max()))
    # Minute samples are averaged; missing samples are never reconstructed.
    table = pd.DataFrame({'gc': gc[eligible].to_numpy()/1000,
                          'pv': pv[eligible].to_numpy()/1000}, index=t[eligible])
    grouped = table.groupby(table.index.floor('30min'))
    means = grouped.mean().reindex(bins)
    counts = grouped.size().reindex(bins, fill_value=0)
    means.loc[counts < 27] = np.nan
    p = np.zeros((366, 48, 3), dtype=np.float32)
    p[:, :, 0] = means['gc'].to_numpy().reshape(366, 48)
    p[:, :, 2] = means['pv'].to_numpy().reshape(366, 48)
    valid = np.isfinite(p[:, :, [0, 2]]).all((1, 2))
    hist = np.zeros(366, dtype=bool)
    for day in range(7, 366):
        hist[day] = valid[day-7:day].all()
    test = valid & hist
    q.update(valid_half_hours=int((counts >= 27).sum()),
             full_30_minute_bins=int((counts == 30).sum()), valid_days=int(valid.sum()),
             external_test_days=int(test.sum()),
             scored_month_counts={str(m): int((test & (date.month == m)).sum()) for m in range(1,13)},
             minimum_valid_bin_minutes=int(counts[counts >= 27].min()),
             maximum_recorded_gc_kw=float(gc.max()/1000), maximum_recorded_pv_kw=float(pv.max()/1000))
    quality.append(q)
    print(q['household_id'], has_pv, q['valid_days'], q['external_test_days'], flush=True)
    if has_pv:
        households.append(record['household']); powers.append(p); valids.append(valid)
        coverages.append(counts.to_numpy().reshape(366,48).astype(np.uint8))

power = np.stack(powers)
valid = np.stack(valids)
history = np.zeros_like(valid)
for day in range(7,366):
    history[:,day] = valid[:,day-7:day].all(axis=1)
test = valid & history
arr = dict(power_kw=power, date=date.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
           household_id=np.asarray(households,dtype=np.int32), valid_gc_gg_day=valid,
           valid_feeder_day=valid.copy(), valid_history=history, external_test_mask=test,
           minute_coverage=np.stack(coverages), source_dataset=np.array('StoreNet original minute W, 2020'),
           channel_names=np.array(['gross_consumption','no_separate_controlled_load','PV_production']))
outputs=[]
for scale in [2.2,2.0,2.4]:
    dest=cache / ('storenet_external_reference_'+str(scale).replace('.','p')+'kwp.npz')
    if dest.exists(): raise FileExistsError(dest)
    np.savez_compressed(dest, **arr, capacity_kwp=np.full(valid.shape,scale,dtype=np.float32))
    outputs.append(dict(path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                        capacity_reference_kwp=scale,bytes=dest.stat().st_size))
audit = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             protocol_path=str(protocol),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
             original_households=len(quality),PV_households=households,
             expected_PV_households_from_paper=10,source_PV_count_agrees=len(households)==10,
             power_shape=list(power.shape), valid_household_days=int(valid.sum()),
             eligible_external_household_days=int(test.sum()),
             joint_all_PV_household_dates=int(test.all(axis=0).sum()),
             pair_mask_sha256=hashlib.sha256(test.tobytes()).hexdigest(),
             outputs=outputs, household_quality=quality)
(research/'storenet_cache_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k != 'household_quality'},indent=2))
