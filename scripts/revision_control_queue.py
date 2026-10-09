"""Revision subprocess queue; completed files retained and failures explicit."""
from revision_forecast_common import *
import subprocess,sys,argparse

def main(args):
    initialize();jobs=[]
    if args.group=='load':
        for load in ['persistence','strong']:
            for model in ['msecont','dfl']:
                tag=f'frozen_{model}_load{load}_tou';jobs.append((None,tag,['--model',model,'--load',load,'--tag',tag]))
            for loss in ['mse','dfl']:
                model=f'matched_pv_{load}_{loss}_tou_wear002';tag=model+'_expanded';jobs.append((REV/f'forecasts/{model}.npz',tag,['--model',model,'--load',load,'--tag',tag]))
    elif args.group=='net':
        for fee in ['tou','flat']:
            tag=f'net_mse_{fee}_seed11_expanded';jobs.append((None,tag,['--model','net_supervised_seed11','--direct-net','--tariff',fee,'--tag',tag]))
    for required,tag,opts in jobs:
        if (REV/f'dispatch/{tag}.json').exists():continue
        waitstart=time.perf_counter()
        while required is not None and (not required.exists() or not (REV/f'results/{required.stem}.json').exists()):
            if time.perf_counter()-waitstart>7200:raise TimeoutError(str(required))
            time.sleep(15)
        with (REV/f'logs/{tag}_queue.log').open('w',encoding='utf-8') as f:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/revision_control_variants.py'),*opts],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Failed {tag}; inspect retained log')
        print('Completed',tag,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['load','net'],required=True);main(ap.parse_args())
