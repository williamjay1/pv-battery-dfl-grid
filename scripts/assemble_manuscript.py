"""Assemble one editable article from audited prose and computed table values."""
from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript'
e=json.loads((M/'article_evidence.json').read_text())
def fmt(q,scale=1,n=3):
    return f'{q["mean"]*scale:.{n}f} [{q["ci95"][0]*scale:.{n}f}, {q["ci95"][1]*scale:.{n}f}]'
tables={}
tables['TABLE1']={
 'caption':'Table 1. Chronological samples and network coverage',
 'headers':['Evaluation unit','Households','Period','Usable observations'],
 'rows':[
 ['Forecast training','244','Jul 2010–Dec 2011','132,200 household-days'],
 ['Forecast validation','244','Jan–Jun 2012','44,373 household-days'],
 ['Known-household forecast test','244','Jul 2012–Jun 2013','88,609 household-days'],
 ['Unseen-household forecast test','55','Jul 2012–Jun 2013','20,032 household-days'],
 ['Network Panel 1','55','Jul 2012–Jun 2013','219 jointly clean days'],
 ['Network Panel 2','55','Jul 2012–Jun 2013','129 jointly clean days'],
 ['Network Panel 3','55','Jul 2012–Jun 2013','276 jointly clean days'],
 ['Unseen-household network panel','55','Jul 2012–Jun 2013','274 jointly clean days'],
 ['Postcode sensitivity','27','Jul 2012–Jun 2013','340 jointly clean days']],
 'note':'Forecast observations require valid targets and histories. Network dates additionally require all participating households and required feeder channels to satisfy the fixed quality rules. The three primary panels are disjoint subsets of the known-household group. Forecast and network denominators differ.'}
assert len(e['main'])==9
tables['TABLE2']={'caption':'Table 2. DFL minus matched MSE continuation',
 'headers':['Panel','Deployment','Clean days','Cost difference (cents/owner-day)','Screening difference (pp)'],
 'rows':[[str(r['panel']+1),'28 ToU' if r['battery_count']==28 else ('55 ToU' if r['tariff_mix']=='all_tou' else '55 mixed'),str(r['voltage_pp']['delta_dfl_minus_mse']['n_clean_days']),fmt(r['cost_owner_cents']),fmt(r['voltage_pp']['delta_dfl_minus_mse'])] for r in e['main']],
 'note':'Values are means and 95% paired calendar-block intervals. Three training seeds are averaged before resampling; 2,000 replicates use seven-day blocks. Cost is AUD cents per battery owner per observed day. Screening is the percentage of all 55 customer half-hours outside 0.95–1.05 pu. ToU denotes time-of-use; mixed denotes 28 ToU and 27 flat-price owners. The physically duplicated 28-battery tariff bookkeeping rows are excluded.'}
names={'none':'No battery','persistence':'Previous-day PV','hgb':'Gradient boosting','mlp':'Frozen MSE','msecont':'Continued MSE','calibrated':'Three-offset calibration','dfl':'Decision focused','stochastic':'20-scenario controller','oracle':'Perfect information','self':'Self-consumption rule'}
c={(r['panel'],r['model']):r for r in e['competitors']}
tr=[]
for m in names:
    a,b=c[0,m],c[3,m]
    rmse='—' if m in ['none','oracle','self','stochastic'] else f'{a["pv_rmse_kw"]:.4f}'
    tr.append([names[m],f'{a["operating_cost_aud_per_household_day"]:.4f}',f'{a["retail_electricity_bill_aud_per_household_day"]:.4f}',f'{a["energy_throughput_kwh_per_household_day"]:.3f}',rmse,f'{b["operating_cost_aud_per_household_day"]:.4f}'])
tables['TABLE3']={'caption':'Table 3. Household operating outcomes under uniform time-of-use prices',
 'headers':['Method','Panel 1 cost (AUD/day)','Panel 1 retail bill (AUD/day)','Throughput (kWh/day)','PV RMSE (kW)','Unseen cost (AUD/day)'],
 'rows':tr,'note':'All values are per household on paired network coverage: 219 days for Panel 1 and 274 for unseen households. Frozen MSE, continued MSE and DFL average seeds 11, 23 and 47; calibration and scenario control use the fixed seed-11 predictor. Operating cost includes retail settlement, controlled load and the 0.02 AUD/kWh throughput proxy. Retail bill already includes controlled load. Throughput and RMSE refer to Panel 1. Scenario forecasts are not represented by a single-RMSE claim. Perfect information is not deployable. The self-consumption reference has a different terminal-state policy.'}
rbs=[r for r in e['robustness'] if r['folder']=='external_full_physical']
main={r['model']:r['outcomes'] for r in e['strong_network']}
r4=[['European LV / 1.050',f'{main["mlp"]["voltage_percentage_points"]:.4f}',f'{main["dfl"]["voltage_percentage_points"]:.4f}',f'{main["dfl"]["voltage_percentage_points"]-main["mlp"]["voltage_percentage_points"]:.4f}',f'{main["dfl"]["daily_import_peak_kw"]-main["mlp"]["daily_import_peak_kw"]:.3f}']]
for r in sorted(rbs,key=lambda r:(r['parameters']['network'],r['parameters']['source_pu'])):
    p=r['parameters'];o=r['outcomes'];v=o['voltage_percentage_points']
    name={'eulv':'European LV','simbench_rural1':'SimBench rural1','simbench_semiurb4':'SimBench semiurb4'}[p['network']]
    r4.append([f'{name} / {p["source_pu"]:.3f}',f'{v["baseline_mean"]:.4f}',f'{v["dfl_mean"]:.4f}',f'{v["delta"]["mean"]:.4f}',f'{o["daily_import_peak_kw"]["delta"]["mean"]:.3f}'])
tables['TABLE4']={'caption':'Table 4. Source-voltage and topology comparisons with frozen MSE',
 'headers':['System / source pu','MSE screening (%)','DFL screening (%)','Screening difference (pp)','Import-peak difference (kW)'],
 'rows':r4,'note':'Panel 1, seed 11, 55 batteries, uniform ToU, and 219 jointly clean dates. Differences are DFL minus frozen MSE; these checks do not control MSE continuation. Voltage exposure uses 0.95–1.05 pu and all 55 customer assignments. Import peak is the mean of daily maxima. Balanced SimBench evaluations and the unbalanced European feeder are separate system conditions.'}
ac=(M/'abstract_conclusion.txt').read_text(encoding='utf-8')
head,tail=ac.split('6. Conclusions',1)
intro=(M/'introduction_related_work.txt').read_text(encoding='utf-8')
method=(M/'methods_draft.txt').read_text(encoding='utf-8')
needle='Missing dates remain present on the 365-day test axis.'
method=method.replace(needle,'Table 1 summarizes the chronological samples and stricter network coverage.\n\n[[TABLE1]]\n\n'+needle)
text=head.strip()+'\n\n'+intro.strip()+'\n\n'+method.strip()+'\n\n'+(M/'results_draft.txt').read_text(encoding='utf-8').strip()+'\n\n'+(M/'discussion_draft.txt').read_text(encoding='utf-8').strip()+'\n\n6. Conclusions'+tail
# Expand authoring reference ranges; assign final numbers by first appearance.
text=re.sub(r'\[L(\d{2})[–-]L(\d{2})\]',lambda m:'['+','.join(f'L{i:02d}' for i in range(int(m[1]),int(m[2])+1))+']',text)
order=[]
for k in re.findall(r'L\d{2}',text):
    if k not in order:order.append(k)
reftext=(M/'references_verified.txt').read_text(encoding='utf-8')
refs={m[1]:m[2].strip() for m in re.finditer(r'\[(L\d{2})\]\s*(.*?)(?=\n\n\[L|\Z)',reftext,re.S)}
assert all(k in refs for k in order)
numbers={k:i+1 for i,k in enumerate(order)}
text=re.sub(r'L\d{2}',lambda m:str(numbers[m[0]]),text)
figures={
 'FIGURE1':{'path':str(M/'figures/figure1_paired_effects.png'),'caption':'Figure 1. Paired household and feeder differences','note':'DFL minus matched MSE continuation under the fixed nighttime restriction. Points are three-seed mean paired differences; bars are 95% intervals from 2,000 seven-day calendar-block resamples. Panels contain 219, 129 and 276 clean dates. Negative values denote lower modeled operating cost, voltage-screening exposure or mean daily import peak. Cost uses battery owners; network metrics use all 55 customers.'},
 'FIGURE2':{'path':str(M/'figures/figure2_signal_and_actions.png'),'caption':'Figure 2. PV signals and battery actions','note':'Panel 1, seed 11, 55 ToU batteries and 219 jointly clean days. Left: mean PV signals and metered generation across households and dates. Right: mean aggregate battery power across dates; positive values denote charging. The common correction is the counterfactual signal intervention relative to continued MSE. The nighttime mask covers fixed windows, not all seasonally varying dark hours.'}}
blocks=[]
for i,p in enumerate(re.split(r'\n\s*\n',text.strip())):
    p=p.strip()
    if not p:continue
    m=re.fullmatch(r'\[\[(TABLE\d|FIGURE\d)\]\]',p)
    if m:
        key=m[1];blocks.append({'type':'table' if key.startswith('TABLE') else 'figure',**(tables.get(key) or figures[key])});continue
    typ='title' if i==0 else ('heading' if re.match(r'^\d[. ]',p) and '\n' not in p and len(p)<110 or p in ['Abstract','Data and computational availability','Declaration of generative AI use'] else 'paragraph')
    if p.startswith(('Delta G =','L_DFL(','e_t =','J_hat =','R(c,d,e) =','0 ≤','1 ≤','e_0 =')):typ='equation'
    blocks.append({'type':typ,'text':p})
blocks.append({'type':'heading','text':'References'})
for k in order:blocks.append({'type':'reference','text':f'[{numbers[k]}] '+refs[k]})
out={'title':blocks[0]['text'],'target_journal':'Applied Energy','stage':'complete research manuscript draft, not an online submission','blocks':blocks,'reference_map':numbers,'author_metadata':'Names, affiliations, funding, competing interests and final author approval must be supplied by the authors before submission.'}
(M/'article_blocks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
review=[]
for b in blocks:
    if b['type'] in ['table','figure']:
        review.append(b['caption'])
        if b['type']=='table':review.extend([' | '.join(b['headers'])]+[' | '.join(row) for row in b['rows']])
        review.append(b['note'])
    else:review.append(b['text'])
(M/'article_full.txt').write_text('\n\n'.join(review),encoding='utf-8')
print(json.dumps({'words':len(' '.join(review).split()),'references':len(order),'tables':len(tables),'figures':len(figures),'body_words_excluding_tables_refs':len(text.split())}))
