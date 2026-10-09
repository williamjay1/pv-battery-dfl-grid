"""Reoptimization under predeclared load inputs and wear coefficients."""
from revision_forecast_common import *
from revision_forecast_train import selected_load
import argparse

def main(args):
    if args.tag and (REV/f'dispatch/{args.tag}.npz').exists() and (REV/f'dispatch/{args.tag}.json').exists():
        old=json.loads((REV/f'dispatch/{args.tag}.json').read_text())
        if old.get('failures')==[]:
            print('Retaining completed revision-only dispatch:',args.tag,flush=True);return
    initialize();a=source_arrays();load=selected_load(a,args.load);battery=Battery(throughput=args.wear)
    if args.model in ('msecont','dfl'):
        pred=old_signal(a,args.model,args.tariff);net=None
    else:
        pred=np.load(REV/f'forecasts/{args.model}.npz')['prediction_kw'];net=pred if args.direct_net else None
        if args.direct_net:pred=None
    name=args.tag or f'{args.model}_load{args.load}_wear{str(args.wear).replace(".","")}_{args.tariff}'
    dispatch(a,name,args.tariff,pv_signal=pred,net_signal=net,load_signal=load,battery=battery,limit_pairs=args.pilot or None)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--direct-net',action='store_true');ap.add_argument('--load',choices=['hgb','persistence','strong'],default='hgb');ap.add_argument('--tariff',choices=['tou','flat'],default='tou');ap.add_argument('--wear',type=float,default=.02);ap.add_argument('--tag');ap.add_argument('--pilot',type=int,default=0);main(ap.parse_args())
