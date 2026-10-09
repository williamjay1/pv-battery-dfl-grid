"""Record a revision protocol and freeze pre-existing model artefacts.

This is a local timestamped record, not a public preregistration.
"""
from pathlib import Path
import hashlib, json
from datetime import datetime, timezone

ROOT = Path('D:/MLWork/pv_battery_dfl_grid_20261005')
REV = ROOT / 'revision_20261009'

items = [
 ('R01','Reframe contribution as method selection under operating conditions','root','manuscript'),
 ('R02','Separate matched objective attribution from practical control comparison','root,forecast_impl','matched cost table'),
 ('R03','Pair continued MSE, DFL, low dimensional calibration and SAA on identical seed and masks','forecast_impl','expanded and feeder economics'),
 ('R04','Report retail bill, throughput proxy, information-gap closure and practical effect size','forecast_impl,root','cost decomposition'),
 ('R05','Measure comparable isolated-process single-thread runtime and peak RSS; explain maintenance','forecast_impl','engineering resource table'),
 ('R06','Use continued MSE primary in source voltage, topology and mapping analyses','network_impl','paired.json'),
 ('R07','Separate over/undervoltage, zero-inclusive and conditional severity; report quantiles, worst node, duration','network_impl','network_diagnostics'),
 ('R08','Predefine complete voltage threshold curves and examine density near thresholds','network_impl,root','voltage figure'),
 ('R09','Report relative peaks/losses, transformer loading and PV/battery/load system scale','network_impl','system table'),
 ('R10','Assess limited, physically interpretable PF/impedance uncertainty','network_impl','physical brackets'),
 ('R11','Continuous SOC with terminal energy value: frozen transfer and matched retraining separately','forecast_impl','control results'),
 ('R12','Officially sourced export rule equally applied and retail repricing; state protection-layer transfer','network_impl,literature_full','export results'),
 ('R13','Direct net-load baseline and training-only seasonal/time calibration test architectural necessity','forecast_impl','baseline results'),
 ('R14','Deployable load models: frozen PV-signal transfer versus newly matched training','forecast_impl','input dependence'),
 ('R15','Describe outputs as controller task signals; retain physical PV diagnostics without claiming improved physics','root','title, methods, discussion'),
 ('R16','AC injection-to-voltage sensitivity with full AC error; temporal/node/phase transmission chain','network_impl,root','mechanism figure'),
 ('R17','Household benefit/harm tails and owner/nonowner voltage exposure','network_impl,forecast_impl','distribution results'),
 ('R18','Retain test-triggered redesign disclosure; freeze existing models before external evaluation','root','external freeze manifest'),
 ('R19','Independent public dataset: verify units, quality, gross load, capacity and frozen-transfer limitations','literature_full,forecast_impl','external results'),
 ('R20','Private economics on all paired valid household-days, separate from feeder complete dates','forecast_impl','expanded sample results'),
 ('R21','Monthly coverage and retained-day characteristics; bootstrap does not correct missingness','forecast_impl,network_impl','coverage table'),
 ('R22','Same 27 dispersed versus geographically grouped households, load/PV matching, same nodes/dates','network_impl','geographic matched check'),
 ('R23','Wear: frozen-action repricing separated from reoptimization and matched retraining at 0/.01/.04','forecast_impl,literature_full','wear results'),
 ('R24','Consolidate evidence boundaries; relocate numerical implementation audits to supplement','root','main and supplement'),
 ('R25','Create minimal executable reproduction package with masks/checkpoints/dispatch/figure data','root','local package'),
 ('R26','Accurate AI disclosure and actual computational verification, without claiming human author signoff','root','declarations'),
 ('R27','Verify journal scope, indexing and nonmandatory OA; resolve JCR Q1/Q2 versus Electrical Engineering','literature_full,root','venue decision'),
 ('R28','Nature-style evidence architecture, editable vector and 1200 dpi raster; rendered alignment/collision and visual audits','root','figures and QA'),
 ('R29','Accessible editable tables, equations and source references, concise substantive prose','root','revised article'),
]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

if __name__ == '__main__':
    (REV/'research').mkdir(exist_ok=True,parents=True)
    route={'route':'SCI','reason':'Applied engineering empirical evaluation of residential storage control and AC feeder consequences.',
           'primary_standard':'Meaningful engineering choice, matched comparisons, information timing, physical assumptions, operational validation and reproducibility.',
           'do_not_use':'Do not require econometric causal identification or claim field deployment, measured degradation or a new DFL algorithm.',
           'next_workflow':'sci-full-workflow','venue':'Original JCR Q1/Q2 nonmandatory OA requirement retained pending user direction; Applied Energy provisional; Electrical Engineering JCR Q2 not verified.'}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'route_card':route,
      'main_question':'When does cost-trained control signalling add practically relevant private and feeder value over matched statistical training, train-only calibration and scenario control?',
      'interpretation':'Conditional computational evidence and selection boundaries, not a universal ranking or physical PV-information improvement.',
      'status':[{'id':a,'requirement':b,'owner':c,'evidence':d,'state':'in_progress'} for a,b,c,d in items]}
    p=REV/'research'/'revision_protocol.json'
    if not p.exists(): p.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    wanted=[ROOT/'checkpoints'/n for n in ['forecast_gc_shared.joblib','forecast_mlp_seed11.pt','forecast_mlp_seed23.pt','forecast_mlp_seed47.pt','decision_physical_tou_seed11.pt','decision_physical_tou_seed23.pt','decision_physical_tou_seed47.pt','mse_continuation_physical_tou_seed11.pt','mse_continuation_physical_tou_seed23.pt','mse_continuation_physical_tou_seed47.pt']]
    wanted += [ROOT/'scripts'/n for n in ['forecast_models.py','battery_model.py','decision_layer.py','train_decision_physical.py','train_matched_mse_physical.py','run_physical_dispatch.py','decision_baselines_physical.py']]
    wanted += [ROOT/'research'/'executed_protocol_final.json',ROOT/'datasets'/'splits_v1.json']
    manifest={'recorded_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Pre-existing Oct6 selected nighttime-restricted Ausgrid models and controller definitions before external outcome inspection; not public preregistration.',
      'external_primary_reference_capacity_kwp':2.2,
      'external_capacity_sensitivity_kwp':[2.0,2.4],
      'external_metadata_caveat':'StoreNet nominal PV capacity differs across paper and repository. Reference scales are not verified nameplate values.',
      'artifacts':[{'relative_path':str(q.relative_to(ROOT)).replace('\\','/'),'bytes':q.stat().st_size,'sha256':digest(q)} for q in wanted if q.exists()]}
    freeze=REV/'research'/'external_model_freeze.json'
    if not freeze.exists(): freeze.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'protocol':str(p),'freeze':str(freeze),'frozen_artifacts':len(manifest['artifacts'])}))
