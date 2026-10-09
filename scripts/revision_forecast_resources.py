"""Assemble completed paired resource measurements without inventing missing runs."""
from revision_forecast_common import *

def main():
    rows={}
    for method in ['msecont','dfl','calibrated3','stochastic20']:
        p=REV/f'results/benchmark_{method}_tou.json'
        if not p.exists():continue
        b=json.loads(p.read_text());row={k:b[k] for k in ['mean_qp_ms','median_qp_ms','p95_qp_ms','mean_process_cpu_qp_ms']};row['inference_peak_rss_mb']=b['peak_process_rss_bytes']/1e6
        if method in ('msecont','dfl'):
            loss='mse' if method=='msecont' else 'dfl';r=json.loads((REV/f'results/matched_pv_hgb_{loss}_tou_wear002.json').read_text());row['training_seconds_including_validation_selection']=r['process_resources']['elapsed_seconds'];row['training_peak_rss_mb']=r['process_resources']['peak_process_rss_bytes']/1e6;row['selected_epoch']=r['candidates'][r['selected']]['selected_epoch'];row['learning_rate']=r['candidates'][r['selected']]['learning_rate'];row['training_days']=r['train_days'];row['validation_days']=r['validation_days'];row['training_scope']='incremental from identical frozen original MSE warm-start; 2 learning rates x 4 epochs, validation cost checkpoint selection, epoch0 eligible';row['update_requirement']='refit neural signal when controller/input economics change'
        elif method=='calibrated3':
            p=REV/'results/calibrated3_tou_seed11.json'
            if p.exists():
                r=json.loads(p.read_text());row['training_seconds_including_validation_selection']=r['training_resources']['elapsed_seconds'];row['training_peak_rss_mb']=r['training_resources']['peak_process_rss_bytes']/1e6;row['training_days']=r['train_days'];row['validation_days']=r['validation_days'];row['training_scope']='3 trainable offsets, 3 epochs, validation shrinkage among five fixed strengths';row['update_requirement']='refit 3 offsets when cost/input setting changes'
        else:
            row['training_seconds_including_validation_selection']=b['preparation_seconds'];row['training_scope']='frozen 20 residual paths: no iterative task-specific training; residual extraction excludes shared MSE/load pretraining';row['update_requirement']='refresh residual bank when distribution changes';row['training_peak_rss_mb']=None
        rows[method]=row
    out={'status':'complete' if len(rows)==4 and 'training_seconds_including_validation_selection' in rows.get('calibrated3',{}) else 'partial','fee':'tou','resource_hardware':'same Windows CPU, deterministic single-thread per process','benchmark_pairs':1024,'benchmark_repeats':3,'shared_cost':'All four methods consume the same pre-existing supervised MSE PV and HGB load models; their original fitting cost is shared and excluded from incremental training time. Continued MSE/DFL resource refits reproduce the frozen procedure but performance uses the original frozen checkpoints.','walltime_caution':'Method benchmark subprocesses were serial relative to one another; unrelated revision experiments ran concurrently. Report measured wall times with this caveat, not an isolated-machine speed claim. CPU-time means supplied but Windows clock granularity affects individual-call CPU times.','peak_memory_caution':'Process peak working set includes interpreter, libraries and loaded prediction/data arrays; it is not model size alone.','rows':rows};write_json(REV/'results/resource_comparison_tou.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
