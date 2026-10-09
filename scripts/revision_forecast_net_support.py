"""Fixed night projection of the already-trained direct-net MLP; no refitting."""
from revision_forecast_common import *

def main():
    initialize();start=time.perf_counter();z=np.load(REV/'forecasts/net_supervised_seed11.npz');pred=z['prediction_kw'].copy();pred[:,:,NIGHT]=np.maximum(pred[:,:,NIGHT],0);np.savez_compressed(REV/'forecasts/net_supervised_night_seed11.npz',household_id=z['household_id'],date=z['date'],prediction_kw=pred);write_json(REV/'results/net_supervised_night_seed11.json',{'method':'net_mse_night','foundation_model':'net_supervised_seed11, unchanged weights','projection':'max(net,0) at slots0:8 and42:48; daytime signed net unchanged','fitting':'No additional fitting or test-based model selection','chronology':'Physical-support correction introduced after inspecting signed-net support during revision; all original results retained',**runtime_info(start)});a=source_arrays()
    for fee in ['tou','flat']:dispatch(a,f'net_mse_night_{fee}_seed11_expanded',fee,net_signal=pred)

if __name__=='__main__':main()
