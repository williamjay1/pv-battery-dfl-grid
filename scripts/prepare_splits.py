"""Freeze temporal/entity splits before model fitting. No outcome-based selection."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def build_splits(root=ROOT):
    p = np.load(root/'datasets/ausgrid_source_clock_panel.npz')
    ids = p['household_id']; dates = p['date']
    # Customer 2 was excluded in the data protocol before forecast fitting.
    eligible = [int(i) for i in ids if i != 2]
    rank = sorted(eligible, key=lambda i: hashlib.sha256(f'ausgrid-split-v1:{i}'.encode()).hexdigest())
    heldout = rank[:55]
    known = rank[55:]
    original_panels = [known[55*j:55*(j+1)] for j in range(3)]
    # Before any model was fitted, household 27's pretest CL incompleteness made
    # an original panel unusable. Apply the same pretest availability rule to all.
    pretest = dates < '2012-07-01'
    rates = p['valid_feeder_day'][:,pretest].mean(axis=1)
    network_eligible = [h for h in known if rates[h-1] >= .95]
    if len(network_eligible) < 165: raise ValueError('Insufficient pretest-qualified known households')
    panels = [network_eligible[55*j:55*(j+1)] for j in range(3)]
    temporal = np.select([dates <= '2011-12-31', dates <= '2012-06-30'], [0, 1], default=2)
    # Day-ahead histories use only the preceding seven source-clock dates.
    valid = p['valid_gc_gg_day']
    history = np.zeros(valid.shape, dtype=bool)
    finite_history = np.zeros(valid.shape, dtype=bool)
    finite = np.isfinite(p['power_kw'][:,:,:,[0,2]]).all(axis=(2,3))
    for d in range(7,len(dates)):
        history[:,d] = valid[:,d-7:d].all(axis=1)
        finite_history[:,d] = finite[:,d-7:d].all(axis=1)
    counts = {}
    for name, group in [('known',known),('heldout',heldout)]:
        hi = np.isin(ids,group)
        counts[name] = {label:int((hi[:,None] & (temporal==k)[None,:] & history & valid).sum()) for k,label in enumerate(['train','validation','test'])}
    def panel_coverage(pp):
        coverage=[]
        for panel in pp:
            h=np.flatnonzero(np.isin(ids,panel))
            clean=(history[h]&p['valid_feeder_day'][h]).all(axis=0)
            coverage.append({name:int((clean&(temporal==k)).sum()) for k,name in enumerate(['train','validation','test'])})
        return coverage
    coverage=panel_coverage(panels)
    def clean_dates(pp):
        result=[]
        for panel in pp:
            h=np.flatnonzero(np.isin(ids,panel))
            clean=(history[h]&p['valid_feeder_day'][h]).all(axis=0)
            result.append({name:dates[clean&(temporal==k)].tolist() for k,name in [(1,'validation'),(2,'test')]})
        return result
    heldout_network=[h for h in heldout if rates[h-1]>=.95]
    result={
      'version':'splits_v1','selection_rule':'sha256(UTF8 ausgrid-split-v1:<customer ID>), ascending; first 55 heldout; next 165 grouped in three disjoint 55-household panels',
      'excluded_ids':[2], 'heldout_household_ids':heldout,'known_household_ids':known,'primary_panels':panels,
      'unused_for_primary_network_but_used_for_pooled_training':[h for h in known if h not in sum(panels,[])],
      'network_eligibility_rule':'At least 95% valid feeder-channel days during training+validation only; apply before fitting, retain original known-household hash order; 244 training households unchanged',
      'network_eligibility_excluded_known_ids':[h for h in known if h not in network_eligible],
      'eligibility_revision_disclosure':{'timing':'Before any forecast fit','reason':'Pretest-incomplete controlled-load channel for household 27 made an original panel unusable','original_panel_common_clean_day_counts':panel_coverage(original_panels),'test_coverage_inspected_for_operational_feasibility':True,'forecast_and_dispatch_results_seen':False},
      'train':['2010-07-01','2011-12-31'],'validation':['2012-01-01','2012-06-30'],'test':['2012-07-01','2013-06-30'],
      'minimum_history_days':7,'train_requires_clean_history_and_target':True,
      'static_capacity_check':'No household changes its nonmissing generator capacity over the supplied source files. Features use preceding-day capacity, never future capacity.',
      'sample_counts_clean_household_days':counts,'panel_common_clean_day_counts':coverage,
      'primary_panel_common_clean_dates':clean_dates(panels),
      'heldout_network_household_ids':heldout_network,
      'heldout_network_common_clean_dates':clean_dates([heldout_network])[0],
      'heldout_network_common_clean_day_counts':panel_coverage([heldout_network])[0],
      'split_frozen_without_reading_forecast_or_dispatch_outcomes':True,
    }
    dest=root/'datasets/splits_v1.json'
    if dest.exists():
        previous=json.loads(dest.read_text(encoding='utf-8'))
        if previous!=result:
            if previous['known_household_ids']==known and previous['heldout_household_ids']==heldout and ('network_eligibility_rule' not in previous or previous['primary_panels']==panels):
                dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
            else: raise FileExistsError('Existing frozen split differs; do not overwrite')
    else: dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=ROOT)
    a=ap.parse_args(); print(json.dumps(build_splits(a.root),indent=2))
