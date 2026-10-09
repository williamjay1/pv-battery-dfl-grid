"""Parallel independent dispatch runs; child outputs remain separate."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import subprocess,sys,os,json
ROOT=Path(__file__).resolve().parents[1]
def work(job):
    model,tariff,seed=job
    path=ROOT/f'results/dispatch/{model}_{tariff}_seed{seed}_test_tight.json'
    if path.exists() and json.loads(path.read_text())['n_failed']==0:return (job,'already complete')
    log=ROOT/f'results/baseline_tight_{model}_{tariff}_{seed}.log'
    cmd=[sys.executable,str(ROOT/'scripts/evaluate_dispatch.py'),'--model',model,'--tariff',tariff,'--seed',str(seed),'--suffix','tight']
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    return (job,r.returncode,str(log))
if __name__=='__main__':
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    jobs=[('mlp',t,s) for t in ['tou','flat'] for s in [11,23,47]]+[(m,t,11) for m in ['none','oracle','hgb','persistence','self'] for t in ['tou','flat']]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in as_completed([pool.submit(work,j) for j in jobs]):print(r.result(),flush=True)
