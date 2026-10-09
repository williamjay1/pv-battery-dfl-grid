"""Execute the frozen learned-policy dispatch comparisons, three jobs at once."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,subprocess,sys,os,json
ROOT=Path(__file__).resolve().parents[1]
def work(job):
    model,tariff,seed=job;dest=ROOT/f'results/dispatch/{model}_{tariff}_seed{seed}_test_tight.json'
    if dest.exists() and json.loads(dest.read_text())['n_failed']==0:return {'job':job,'status':'reused'}
    log=ROOT/f'results/dispatch_{model}_{tariff}_{seed}_tight.log'
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_dispatch.py'),'--model',model,'--tariff',tariff,'--seed',str(seed),'--suffix','tight']
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    if r.returncode:raise RuntimeError(f'{job} failed: {log}')
    return {'job':job,'status':'complete','log':str(log)}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['dfl']);args=ap.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[(m,t,s) for m in args.models for t in ['tou','flat'] for s in [11,23,47]]
        for f in as_completed([pool.submit(work,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
