"""Nature-style figures from measured revision outputs (no demonstration values).

Visual conventions: SciencePlots + nature-figure v2.8.0 (Apache-2.0),
https://github.com/Yuan1z0825/nature-skills @ 2a20e4a.
Statistical calculations remain in the study analysis code.
"""
from pathlib import Path
import os, sys, json, argparse, hashlib
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'revision_20261009'; OUT=REV/'figures'
os.environ.setdefault('MPLCONFIGDIR',str(REV/'temp'/'matplotlib'))
sys.path.insert(0,r'C:\Users\Administrator\.agents\skills\sci-figures')
VENDOR=REV/'vendor'/'Yuan1z0825-nature-skills-2a20e4a'/'skills'/'nature-figure'/'scripts'
sys.path.insert(0,str(VENDOR))
from figstyle import apply_house_style, audit, mm
from audit_panel_alignment import require_matplotlib_panel_alignment
import numpy as np
plt=apply_house_style(styles=('science','nature'),fontsize=7,fonttype=42,use_tex=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
  'pdf.fonttype':42,'svg.fonttype':'none',
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,
  'xtick.direction':'out','ytick.direction':'out','xtick.top':False,'ytick.right':False,
  'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.minor.visible':False,'ytick.minor.visible':False,
  'lines.linewidth':1.1,'savefig.bbox':None,'savefig.pad_inches':0,'legend.handlelength':2.2})
BLUE='#285F8F'; TEAL='#27857A'; RED='#B45A52'; GRAY='#555B61'; GOLD='#B88742'
PALETTE={'msecont':GRAY,'dfl':BLUE,'calibrated':TEAL,'seasonal':GOLD,'stochastic':RED}
OUT.mkdir(parents=True,exist_ok=True)

def json_read(p):return json.loads(p.read_text(encoding='utf-8'))
def label(ax,letter,title):
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(-25,10),textcoords='offset points',
      ha='left',va='bottom',fontsize=8,fontweight='bold',annotation_clip=False)
    ax.set_title(title,loc='left',pad=13,fontsize=7,fontweight='normal')

def export(fig,stem,contract,sources):
    p=OUT/stem
    fig.canvas.draw()
    geometry=audit(fig,max_width_mm=180,verbose=False)
    if not geometry['ok']: raise RuntimeError(geometry)
    require_matplotlib_panel_alignment(fig,json_out=str(p)+'.alignment.json',
      overlay_svg=str(p)+'.alignment.svg',require_panel_labels=True,strict=True,
      tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(str(p)+'.pdf',dpi=1200,bbox_inches=None)
    fig.savefig(str(p)+'.svg',dpi=1200,bbox_inches=None)
    fig.savefig(str(p)+'.png',dpi=1200,bbox_inches=None)
    fig.savefig(str(p)+'.tiff',dpi=1200,bbox_inches=None,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(str(p)+'.preview.png',dpi=300,bbox_inches=None)
    provenance={'contract':contract,'sources':sources,'geometry':geometry,
      'vector':'PDF TrueType embedded; SVG text editable','raster_dpi':1200,
      'style_sources':{'SciencePlots':'2.2.2','nature-figure':'2a20e4a0868ef9094257cb5386cfe623454ae092'},
      'numerical_integrity':'All quantities derive from actual output, no image-derived numbers or simulated plot data.'}
    (OUT/(stem+'.provenance.json')).write_text(json.dumps(provenance,indent=2,default=lambda x:x.item()),encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'figure':stem,'geometry':geometry},default=lambda x:x.item()))

def scenario(records,method,network='eulv',source=1.05):
    for r in records:
        m=r['metadata']; p=m['parameters']
        if (m['objective']==method and m['seed']==11 and m['panel']==0
            and m['battery_count']==55 and m['tariff_mix']=='all_tou'
            and m['comparison_group'] in (None,'original_physical')
            and p.get('power_factor',.95)==.95 and p.get('impedance_scale',1.0)==1.0
            and p.get('export_limit_kw') is None
            and p['network']==network and p['source_pu']==source and p.get('mapping_seed') is None):
            return r
    raise KeyError((method,network,source))

def fig1_economics():
    report=REV/'results'/'expanded_comparison_tou_support_matched_baselines.json'
    j=json_read(report); cohort=j['cohorts']['all299']; ref=j['reference']
    specs=[('msecont','Continued MSE',GRAY),('calibrated3','3-offset calibration',TEAL),
      ('seasonal12','12-offset calibration',GOLD),('net_mse_night','Net-load MSE','#9A8CA7'),
      ('net_calibrated24_night','Net-load calibration','#777083'),('dfl','DFL',BLUE),('stochastic20','20 scenarios',RED)]
    fig=plt.figure(figsize=(mm(180),mm(124)))
    gs=fig.add_gridspec(2,2,width_ratios=[1,1],height_ratios=[1,1])
    a=fig.add_subplot(gs[:,0]); b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1])
    fig.subplots_adjust(left=.225,right=.972,bottom=.115,top=.86,wspace=.70,hspace=.88)
    source=[]
    for i,(key,name,color) in enumerate(specs):
        file=f'{key}_tou_seed11_expanded';v=0.;ci=[0.,0.]
        if file!=ref:
            r=cohort['paired'][file+'__minus__'+ref];v=r['micro_mean_difference']*100;ci=np.array(r['micro_calendar_block95'])*100
        a.errorbar(v,i,xerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],fmt='o',color=color,markersize=4,
          capsize=2,elinewidth=.9)
        source.append({'method':key,'difference_cents':v,'ci_cents':list(ci)})
    a.axvline(0,color='#A9ABB0',ls=':',lw=.8)
    a.set(xlabel='Cost change (AUD cents / household-day)',yticks=np.arange(7),yticklabels=[s[1] for s in specs],ylim=(6.6,-.6))
    z0=np.load(REV/'dispatch'/(ref+'.npz'));m=z0['valid_gcgg_day']; baseline=z0['operating_excluding_cl_aud']
    distributions={}; decomposition=[]
    for i,(key,name,col) in enumerate([specs[1],specs[5],specs[6]]):
        file=f'{key}_tou_seed11_expanded';z=np.load(REV/'dispatch'/(file+'.npz'))
        assert np.array_equal(z['valid_gcgg_day'],m)
        diff=z['operating_excluding_cl_aud']-baseline
        hh=np.sum(np.where(m,diff,0),axis=1)/m.sum(axis=1)*100
        sorted_hh=np.sort(hh);cdf=np.arange(1,len(hh)+1)/len(hh)*100
        b.step(sorted_hh,cdf,where='post',color=col,ls=['--','-','-.'][i],label=name)
        distributions[key]={'household_id':z['household_id'].tolist(),'household_mean_cents':hh.tolist()}
        q=cohort['paired'][file+'__minus__'+ref]; retail=q['retail_component_difference']*100;wear=q['wear_component_difference']*100
        c.barh(i,retail,height=.48,color=BLUE,label='Retail settlement' if i==0 else None)
        c.barh(i,wear,left=min(0,retail),height=.48,color='#AEBCC7',label='Throughput proxy' if i==0 else None)
        decomposition.append({'method':key,'retail_cents':retail,'wear_proxy_cents':wear})
    b.axvline(0,color='#A9ABB0',ls=':',lw=.8)
    b.set(xlabel='Household mean cost change (AUD cents)',ylabel='Households (%)',ylim=(0,102))
    b.legend(loc='upper left',bbox_to_anchor=(0,1.43),ncol=3,fontsize=6,columnspacing=.8,handlelength=1.6)
    c.axvline(0,color='#A9ABB0',ls=':',lw=.8)
    c.set(xlabel='Cost component change (AUD cents)',yticks=np.arange(3),yticklabels=['3 offsets','DFL','20 scenarios'],ylim=(2.55,-.55))
    c.legend(loc='upper left',bbox_to_anchor=(0,1.40),ncol=2,fontsize=6,columnspacing=.9,handlelength=1.5)
    label(a,'a','Practical alternatives on paired household-days')
    label(b,'b','Who benefits or loses')
    label(c,'c','Retail and throughput contributions')
    (OUT/'figure1_source_data.json').write_text(json.dumps({'paired_costs':source,'household_distributions':distributions,'components':decomposition},indent=2),encoding='utf-8')
    export(fig,'figure1_economics',{'question':'Does DFL change the practical controller choice relative to low-complexity and scenario alternatives?',
      'claim':'DFL adds a small broadly distributed benefit; scenario control yields lower mean cost and throughput pricing explains a substantial share.',
      'panel_roles':{'a':'decisive paired alternatives, including unsuccessful direct-net and seasonal models','b':'full household benefit/harm distribution','c':'retail versus modelled wear accounting'},
      'sample':'299 households, 108641 paired GC/PV household-days, seed11; CL-invariant contrasts.',
      'uncertainty':'95% seven-day calendar-block intervals, 2000 resamples; household empirical CDF has no population interpretation.'},
      ['results/expanded_comparison_tou_support_matched_baselines.json','dispatch/*_tou_seed11_expanded.npz'])

def fig2_voltage():
    data=REV/'network_diagnostics'
    records=json_read(data/'scenarios.json')['records']; pairs=json_read(data/'fixed_seed11_paired.json')['records']
    conditions=[('eulv',1.05,'LV 1.050'),('eulv',1.025,'LV 1.025'),('eulv',1.,'LV 1.000'),
                ('simbench_rural1',1.025,'Rural 1.025'),('simbench_semiurb4',1.025,'Semiurban 1.025')]
    # Do not silently omit incomplete conditions from a delivery figure.
    for n,s,_ in conditions:
        scenario(records,'msecont',n,s);scenario(records,'dfl',n,s)
    fig,axs=plt.subplots(2,2,figsize=(mm(180),mm(139)))
    fig.subplots_adjust(left=.11,right=.975,bottom=.095,top=.84,wspace=.52,hspace=.66)
    colors={1.05:BLUE,1.025:TEAL,1.:GOLD}; rows=[]
    for s in (1.05,1.025,1.):
        for method,ls in [('msecont','--'),('dfl','-')]:
            r=scenario(records,method,'eulv',s); z=np.load(data/(r['scenario_id']+'.npz'))
            axs[0,0].plot(z['threshold_upper_pu'],z['upper_exposure_pct'],color=colors[s],ls=ls)
            axs[0,1].plot(z['threshold_lower_pu'],z['lower_exposure_pct'],color=colors[s],ls=ls)
            rows += [{'source_pu':s,'method':method,'upper_threshold':float(x),'upper_pct':float(y)} for x,y in zip(z['threshold_upper_pu'],z['upper_exposure_pct'])]
            rows += [{'source_pu':s,'method':method,'lower_threshold':float(x),'lower_pct':float(y)} for x,y in zip(z['threshold_lower_pu'],z['lower_exposure_pct'])]
    axs[0,0].set(xlabel='Upper threshold (pu)',ylabel='Samples above threshold (%)',xlim=(.98,1.10),ylim=(-2,102))
    axs[0,0].set_xticks([.98,1.01,1.04,1.07,1.10])
    axs[0,1].set(xlabel='Lower threshold (pu)',ylabel='Samples below threshold (%)',xlim=(.90,1.02))
    axs[0,1].set_xticks([.90,.93,.96,.99,1.02])
    y=np.arange(len(conditions)); delta_rows=[]; severity=[]
    for k,(net,src,name) in enumerate(conditions):
        q=next(r for r in pairs if r['reference']=='msecont' and r['key'][0]==0 and r['key'][1]==55
               and r['key'][2]=='all_tou' and r['key'][3]==net and r['key'][4]==src and r['key'][5] is None
               and r.get('seeds')==[11] and r.get('comparison_group')=='original_physical'
               and r['key'][6:]==[.95,1,None])
        for key,col,marker,offset in [('daily_over_pct',BLUE,'o',-.13),('daily_under_pct',BLUE,'s',.13)]:
            d=q['outcomes'][key]['delta']; mu=d['mean']; lo,hi=d['ci95']
            axs[1,0].errorbar(mu,k+offset,xerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt=marker,
              color=col,markerfacecolor=col if marker=='o' else 'white',markersize=3.5,capsize=2,elinewidth=.8)
            delta_rows.append({'condition':name,'metric':key,**d})
        for method,offset in [('msecont',-.10),('dfl',.10)]:
            r=scenario(records,method,net,src)
            for direction,marker,shift in [('over','o',-.13),('under','s',.13)]:
                value=r['voltage'][f'conditional_{direction}_pu']
                if value is not None and np.isfinite(value):
                    axs[1,1].plot(value*1000,k+offset+shift,marker,color=PALETTE[method],
                      markerfacecolor=PALETTE[method] if marker=='o' else 'white',markersize=3.5)
                severity.append({'condition':name,'method':method,'direction':direction,'conditional_excess_pu':value})
    axs[1,0].axvline(0,color='#A9ABB0',ls=':',lw=.8)
    axs[1,0].set(xlabel='DFL minus continued MSE (pp)',yticks=y,yticklabels=[q[2] for q in conditions],ylim=(4.5,-.5))
    axs[1,1].set(xlabel='Conditional excursion (0.001 pu)',yticks=y,yticklabels=[q[2] for q in conditions],ylim=(4.5,-.5),xlim=(0,None))
    # Legends occupy allocated whitespace, never scientific marks.
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=colors[s],label=f'Source {s:.3f} pu') for s in (1.05,1.025,1.)]
    handles += [Line2D([],[],color=GRAY,ls='--',label='Continued MSE'),Line2D([],[],color=GRAY,ls='-',label='DFL')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.98),ncol=5,columnspacing=1.4,fontsize=6)
    fig.legend(handles=[Line2D([],[],color=GRAY,marker='o',ls='',label='Overvoltage'),
      Line2D([],[],color=GRAY,marker='s',mfc='white',ls='',label='Undervoltage'),
      Line2D([],[],color=GRAY,marker='o',ls='',label='Continued MSE'),
      Line2D([],[],color=BLUE,marker='o',ls='',label='DFL')],loc='upper center',bbox_to_anchor=(.54,.486),ncol=4,fontsize=6)
    label(axs[0,0],'a','Upper voltage tail across source settings')
    label(axs[0,1],'b','Lower voltage tail across source settings')
    label(axs[1,0],'c','Matched exposure change at 0.95–1.05 pu')
    label(axs[1,1],'d','Magnitude conditional on an excursion')
    (OUT/'figure2_source_data.json').write_text(json.dumps({'threshold_curves':rows,'matched_changes':delta_rows,'conditional_severity':severity},indent=2),encoding='utf-8')
    export(fig,'figure2_voltage',{'question':'Does DFL mitigate a meaningful voltage problem or shift near-threshold counts?',
      'claim':'Matched exposure and conditional severity depend on source voltage and topology; full fixed threshold curves bound a threshold-specific inference.',
      'panel_roles':{'a':'upper-tail distribution','b':'lower-tail distribution','c':'matched topology and source control with uncertainty','d':'conditional magnitude, undefined if no excursion'},
      'undefined_policy':'No exceedance means conditional severity undefined, displayed as absent points rather than zero.',
      'uncertainty':'95% paired seven-day calendar-block intervals, seed11; descriptive threshold curves.'},
      ['network_diagnostics/scenarios.json','network_diagnostics/fixed_seed11_paired.json'])

def fig3_mechanism():
    folder=REV/'network_mechanism'; z=dict(np.load(folder/'first_prespecified_day.npz'))
    for variable in ('charge_kw','discharge_kw'):
        z['delta_'+variable]=z['target_'+variable]-z['baseline_'+variable]
    q=np.load(folder/'arrays.npz'); report=json_read(folder/'summary.json')
    fig,axs=plt.subplots(2,2,figsize=(mm(180),mm(139)))
    fig.subplots_adjust(left=.105,right=.975,bottom=.10,top=.87,wspace=.52,hspace=.72)
    t=z['hour']; phase=z['house_phase']; phasecolors=[BLUE,TEAL,GOLD]
    a,b,c,d=axs.ravel()
    a.plot(t,z['delta_forecast_pv_kw'].mean(axis=1),color=BLUE)
    a.axhline(0,color='#A9ABB0',lw=.6,ls=':')
    a.set(xlabel='Source-clock hour',ylabel='Mean task-signal change (kW)',xticks=[0,6,12,18,24],xlim=(0,24))
    b.plot(t,z['delta_charge_kw'].sum(axis=1),color=BLUE,label='Charging')
    b.plot(t,z['delta_discharge_kw'].sum(axis=1),color=TEAL,label='Discharging')
    b.plot(t,z['delta_net_injection_kw'].sum(axis=1),color=RED,ls='--',label='Net injection')
    b.axhline(0,color='#A9ABB0',lw=.6,ls=':')
    b.set(xlabel='Source-clock hour',ylabel='Feeder aggregate change (kW)',xticks=[0,6,12,18,24],xlim=(0,24))
    b.legend(loc='upper left',bbox_to_anchor=(-.03,1.39),ncol=3,fontsize=6,columnspacing=.8,handlelength=1.4)
    for k,p in enumerate(np.unique(phase)):
        c.plot(t,z['delta_voltage_pu'][:,phase==p].mean(axis=1)*1000,color=phasecolors[k],label=f'Phase {p}')
    c.axhline(0,color='#A9ABB0',lw=.6,ls=':')
    c.set(xlabel='Source-clock hour',ylabel='Mean voltage change (0.001 pu)',xticks=[0,6,12,18,24],xlim=(0,24))
    c.legend(loc='upper left',bbox_to_anchor=(-.03,1.39),ncol=3,fontsize=6,columnspacing=.9,handlelength=1.6)
    actual=(q['target_voltage_pu']-q['baseline_voltage_pu']).ravel()*1000
    linear=q['linear_voltage_delta_pu'].ravel()*1000
    d.scatter(actual,linear,s=8,c=BLUE,alpha=.6,edgecolors='none')
    extent=max(abs(actual).max(),abs(linear).max())*1.10
    d.plot([-extent,extent],[-extent,extent],color=GRAY,ls='--',lw=.8,zorder=0)
    d.set(xlabel='Full AC change (0.001 pu)',ylabel='Local linear change (0.001 pu)',xlim=(-extent,extent),ylim=(-extent,extent))
    label(a,'a','Task-signal adjustment on the first fixed date')
    label(b,'b','Battery action and injection transmission')
    label(c,'c','Phase-resolved continuous voltage change')
    label(d,'d','Full AC check of 19 prescribed conditions')
    source={'date':str(z['date'].item()) if 'date' in z else '2012-08-24','source_clock_hour':t.tolist(),
      'mean_signal_delta_kw':z['delta_forecast_pv_kw'].mean(axis=1).tolist(),
      'charge_delta_kw':z['delta_charge_kw'].sum(axis=1).tolist(),
      'discharge_delta_kw':z['delta_discharge_kw'].sum(axis=1).tolist(),
      'net_injection_delta_kw':z['delta_net_injection_kw'].sum(axis=1).tolist(),
      'phase_mean_voltage_delta_pu':{str(p):z['delta_voltage_pu'][:,phase==p].mean(axis=1).tolist() for p in np.unique(phase)},
      'full_ac_delta_pu':(actual/1000).tolist(),'linear_delta_pu':(linear/1000).tolist(),'cases':report['cases']}
    (OUT/'figure3_source_data.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    export(fig,'figure3_mechanism',{'question':'How does a small private-objective adjustment reach particular network voltages?',
      'claim':'Fixed-date action changes transmit through nodal demand with phase coupling; local sensitivities approximate continuous voltage changes.',
      'sample':'First prescribed date 2012-08-24, 55 sites; 19 prescribed operating points x 55 receiving sites for local AC audit.',
      'selection':'First common clean date per season plus three exogenous net-demand quantiles, no effect-ranking.',
      'sign':'DFL minus continued MSE. Net injection is negative net demand; fixed truth implies delta injection = delta discharge - delta charge.',
      'uncertainty':'Descriptive fixed examples, not independent measurements or population inference.'},
      ['network_mechanism/first_prespecified_day.npz','network_mechanism/arrays.npz','network_mechanism/summary.json'])

def fig4_boundaries():
    q=json_read(REV/'results'/'control_transfer_summary.json')
    if q['status']!='complete':raise RuntimeError('Control summary is incomplete')
    e=json_read(REV/'results'/'storenet_frozen_external.json')
    fig,axs=plt.subplots(2,2,figsize=(mm(180),mm(139)))
    fig.subplots_adjust(left=.17,right=.975,bottom=.10,top=.855,wspace=.60,hspace=.80)
    a,b,c,d=axs.ravel();source=[]
    def point(ax,row,y,color,marker='o'):
        v=row.get('micro_mean_difference_aud',row.get('micro_mean_difference'))*100
        ci=np.array(row['calendar_block95'])*100
        ax.errorbar(v,y,xerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],fmt=marker,color=color,
          markersize=3.7,capsize=2,elinewidth=.8,markerfacecolor=color if marker=='o' else 'white')
        return {'difference_cents':v,'ci_cents':ci.tolist()}
    for i,key in enumerate(('persistence','strong')):
        for category,offset,col in [('frozen',-.13,GRAY),('matched',.13,BLUE)]:
            row=q['load'][key][category]['all299'];source.append({'panel':'a','load':key,'category':category,**point(a,row,i+offset,col)})
    a.set(yticks=[0,1],yticklabels=['Previous day','HGB–MLP mixture'],ylim=(1.6,-.6),xlabel='Cost change (AUD cents / household-day)')
    for i,key in enumerate(('frozen','matched')):
        for measure,offset,col,marker in [('raw_operating_contrast',-.11,GRAY,'s'),('inventory_adjusted_contrast',.11,BLUE,'o')]:
            row=q['continuous'][key][measure]['all299'];source.append({'panel':'b','training':key,'measure':measure,**point(b,row,i+offset,col,marker)})
    b.set(yticks=[0,1],yticklabels=['Frozen signals','Matched retraining'],ylim=(1.6,-.6),xlabel='Cost change (AUD cents / household-day)')
    wearstyles=[('frozen_actions_repriced',GRAY,'s',-.16),('frozen_signals_reoptimized',TEAL,'^',0),('matched_signals_reoptimized',BLUE,'o',.16)]
    for i,key in enumerate(('0','0.01','0.04')):
        for category,col,marker,offset in wearstyles:
            row=q['wear'][key][category]['all299'];source.append({'panel':'c','wear':key,'category':category,**point(c,row,i+offset,col,marker)})
    c.set(yticks=[0,1,2],yticklabels=['0.00','0.01','0.04'],ylabel='Throughput price (AUD / kWh)',ylim=(2.6,-.6),xlabel='Cost change (AUD cents / household-day)')
    for i,key in enumerate(('2p0','2p2','2p4')):
        for fee,offset,col,marker in [('tou',-.12,BLUE,'o'),('flat',.12,GOLD,'s')]:
            row=e['scales'][key]['fees'][fee]['three_seed_mean_paired'];source.append({'panel':'d','reference':key,'tariff':fee,**point(d,row,i+offset,col,marker)})
    d.set(yticks=[0,1,2],yticklabels=['2.0','2.2 (fixed)','2.4'],ylabel='Normalization reference (kWp)',ylim=(2.6,-.6),xlabel='Cost change (AUD cents / household-day)')
    for ax in axs.ravel():ax.axvline(0,color='#A9ABB0',lw=.7,ls=':')
    from matplotlib.lines import Line2D
    def legend(ax,items,ncol):
        ax.legend(handles=[Line2D([],[],marker=m,color=col,ls='',mfc=col if m=='o' else 'white',markersize=3.5,label=name) for name,col,m in items],loc='upper left',bbox_to_anchor=(-.03,1.44),ncol=ncol,fontsize=6,columnspacing=.85,handlelength=1.2)
    legend(a,[('Frozen signals',GRAY,'o'),('Matched retraining',BLUE,'o')],2)
    legend(b,[('Raw operating',GRAY,'s'),('Inventory-adjusted',BLUE,'o')],2)
    legend(c,[('Reprice',GRAY,'s'),('Reoptimize',TEAL,'^'),('Retrain',BLUE,'o')],3)
    legend(d,[('Time-of-use',BLUE,'o'),('Flat',GOLD,'s')],2)
    label(a,'a','Deployable load-input change')
    label(b,'b','Continuous state of charge')
    label(c,'c','Accounting versus policy change')
    label(d,'d','Frozen external-data transfer')
    (OUT/'figure4_source_data.json').write_text(json.dumps({'paired_rows':source},indent=2),encoding='utf-8')
    export(fig,'figure4_boundaries',{'question':'Which operating changes preserve a trained signal, and which require adaptation?',
      'claim':'Frozen persistence transport differs from matched adaptation; wear and external tariff change the size or existence of the benefit.',
      'sample':'108641 Ausgrid household-days, seed11; 2224 StoreNet household-days, three seeds averaged.',
      'uncertainty':'95% seven-day paired calendar-block intervals; all fixed reference capacity scales retained.',
      'flat_freeze_caveat':'Original six flat checkpoints were omitted from pre-run manifest and recorded transparently in post-run addendum.'},
      ['results/control_transfer_summary.json','results/storenet_frozen_external.json'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--figure',choices=['voltage','economics','mechanism','boundaries'],required=True); args=a.parse_args()
    if args.figure=='voltage':fig2_voltage()
    if args.figure=='economics':fig1_economics()
    if args.figure=='mechanism':fig3_mechanism()
    if args.figure=='boundaries':fig4_boundaries()
