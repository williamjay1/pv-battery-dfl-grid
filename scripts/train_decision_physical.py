"""Post-diagnostic physical-support redesign, preserving the audited DFL trainer.

The only factory override is explicit and process-local: train_decision.PVMLP is
replaced by PhysicalPVMLP. The audited train_one/evaluate implementations receive
that model, so masking applies in training, validation and final prediction.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import torch
from forecast_models import PVMLP,load_data,make_features,build_masks
from battery_model import Battery
import train_decision as audited

ROOT=Path(__file__).resolve().parents[1]
NIGHT_SLOTS=list(range(8))+list(range(42,48))
REDESIGN='Introduced after inspection of unconstrained-model test nighttime diagnostics; not preregistered. Original results retained. No test outcome is used for LR/epoch selection in this redesign.'

class PhysicalPVMLP(PVMLP):
    def __init__(self,input_dim=774):
        super().__init__(input_dim)
        mask=torch.ones(48);mask[NIGHT_SLOTS]=0
        self.register_buffer('physical_support_mask',mask,persistent=False)
    def forward(self,x):
        return super().forward(x)*self.physical_support_mask

def frozen_inputs():
    p,splits=load_data();_,finite,_,_,masks=build_masks(p,splits)
    baseline=json.loads((ROOT/'results/decision_training_v1.json').read_text())
    assert baseline['status']=='complete'
    tr=np.asarray(baseline['train_pairs']);va=np.asarray(baseline['validation_pairs'])
    assert masks['train'][tr[:,0],tr[:,1]].all() and masks['validation'][va[:,0],va[:,1]].all()
    assert len(tr)==4096 and len(va)==1024
    return p,finite,baseline,tr,va

def predict_all(model,p,finite):
    result=np.full(p['power_kw'].shape[:3],np.nan,np.float32);pairs=np.argwhere(finite)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(pairs),4096):
            pair=pairs[start:start+4096];h,d=pair.T
            result[h,d]=model(torch.from_numpy(make_features(p,pair))).numpy()*p['capacity_kwp'][h,d-1,None]
    assert np.isfinite(result[finite]).all() and (result[finite][:,NIGHT_SLOTS]==0).all()
    return result

def merge(family):
    output={};reports=[]
    for kind in ['tou','flat']:
        if family=='decision':name=f'decision_arrays_physical_{kind}.npz';report=f'decision_training_physical_{kind}.json'
        else:name=f'mse_continuation_arrays_physical_{kind}.npz';report=f'mse_continuation_training_physical_{kind}.json'
        run=json.loads((ROOT/'results'/report).read_text());assert run['status']=='complete';reports.append(run)
        with np.load(ROOT/'datasets'/name) as z:
            for k in z.files:
                if k in output:assert np.array_equal(output[k],z[k])
                else:output[k]=z[k]
    target=ROOT/'datasets'/f'{family if family=="decision" else "mse_continuation"}_arrays_physical.npz'
    np.savez_compressed(target,**output)
    combined={'status':'complete','model_class':'PhysicalPVMLP','night_zero_slots':NIGHT_SLOTS,
              'redesign_disclosure':REDESIGN,'test_used_for_selection':False,
              'tariff_reports':reports,'array_file':str(target)}
    if family=='decision':
        combined.update(train_pairs=reports[0]['train_pairs'],validation_pairs=reports[0]['validation_pairs'],
                        tuning=[v for r in reports for v in r['tuning']],final=[v for r in reports for v in r['final']])
        report_name='decision_training_physical.json'
    else:
        combined.update(tuning=[v for r in reports for v in r['tuning']],records=[v for r in reports for v in r['records']],
                        selections=[{k:r[k] for k in ['tariff','selected_config','dfl_selected_config','same_lr_equals_primary']} for r in reports])
        report_name='mse_continuation_training_physical.json'
    (ROOT/'results'/report_name).write_text(json.dumps(combined,indent=2))
    print(json.dumps({'status':'merged','file':str(target),'keys':list(output)}),flush=True)

def main(args):
    if args.merge:merge('decision');return
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    kind=args.tariff;p,finite,baseline,tr,va=frozen_inputs();arr=np.load(ROOT/'datasets/forecast_arrays_v1.npz')
    # This explicit model factory injection is local to this new process only.
    audited.PVMLP=PhysicalPVMLP
    configs=[r['config'] for r in baseline['tuning'] if r['tariff']==kind]
    assert len(configs)==2
    tag=f'physical_{kind}';report_path=ROOT/f'results/decision_training_{tag}.json';start=time.perf_counter()
    report={'status':'running','tariff':kind,'model_class':'PhysicalPVMLP','model_base':'PVMLP 128-64-48',
            'night_zero_slots':NIGHT_SLOTS,'night_zero_in_training_validation_inference':True,'redesign_disclosure':REDESIGN,
            'trainer':'Unmodified train_decision.train_one with process-local PhysicalPVMLP factory',
            'audited_trainer_sha256':hashlib.sha256((ROOT/'scripts/train_decision.py').read_bytes()).hexdigest(),
            'source_pairs':'decision_training_v1.json','train_pairs':tr.tolist(),'validation_pairs':va.tolist(),
            'test_used_for_selection':False,'tuning':[],'final':[]}
    runs=[]
    for cfg in configs:
        model,log=audited.train_one(p,arr,tr,va,11,kind,cfg,Battery());runs.append((model,log));report['tuning'].append(log)
        report_path.write_text(json.dumps(report,indent=2))
    winner=int(np.argmin([r[1]['validation_best_bill'] for r in runs]));out={'household_id':p['household_id'],'date':p['date']}
    for seed in [11,23,47]:
        model,log=runs[winner] if seed==11 else audited.train_one(p,arr,tr,va,seed,kind,runs[winner][1]['config'],Battery())
        report['final'].append(log)
        torch.save({'model_state_dict':model.state_dict(),'model_class':'PhysicalPVMLP','night_zero_slots':NIGHT_SLOTS,'seed':seed,'tariff':kind,'log':log,'redesign_disclosure':REDESIGN},ROOT/f'checkpoints/decision_physical_{kind}_seed{seed}.pt')
        out[f'pv_dfl_{kind}_seed{seed}']=predict_all(model,p,finite)
        report_path.write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/f'datasets/decision_arrays_{tag}.npz',**out)
    report.update(status='complete',elapsed_seconds=time.perf_counter()-start)
    report_path.write_text(json.dumps(report,indent=2));print('COMPLETE '+tag,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tariff',choices=['tou','flat']);ap.add_argument('--merge',action='store_true');args=ap.parse_args()
    if not args.merge and args.tariff is None:ap.error('--tariff required unless --merge')
    main(args)
