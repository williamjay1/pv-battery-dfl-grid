"""New physically supported regime; never replace the original tight results."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,subprocess,sys,os,json
ROOT=Path(__file__).resolve().parents[1]
def work(job):
    model,tariff,seed=job;dest=ROOT/f'results/dispatch/{model}_{tariff}_seed{seed}_test_physical.json'
    if dest.exists() and json.loads(dest.read_text()).get('n_failed',1)==0:return {'job':job,'status':'reused'}
    log=ROOT/f'results/dispatch_{model}_{tariff}_{seed}_physical.log'
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_dispatch.py'),'--model',model,'--tariff',tariff,'--seed',str(seed),'--suffix','physical','--decision-tag','physical','--mse-tag','physical']
    if model not in ['none','self','oracle']:cmd+=['--night-zero']
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    if r.returncode:raise RuntimeError(f'{job} failed: {log}')
    return {'job':job,'status':'complete','log':str(log)}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['baselines','learned'],default='baselines');args=ap.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.stage=='baselines':jobs=[('mlp',t,s) for t in ['tou','flat'] for s in [11,23,47]]+[(m,t,11) for m in ['none','hgb','persistence','oracle','self'] for t in ['tou','flat']]
    else:jobs=[(m,t,s) for m in ['dfl','msecont'] for t in ['tou','flat'] for s in [11,23,47]]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(work,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
