"""Audit the recorded, post-diagnostic night support correction; no fitting."""
from revision_forecast_common import *
from revision_forecast_calibration import design


def main():
    initialize(); start = time.perf_counter(); a = source_arrays()
    raw = np.load(REV/'forecasts/net_supervised_seed11.npz')['prediction_kw']
    base = np.load(REV/'forecasts/net_supervised_night_seed11.npz')['prediction_kw']
    expected = raw.copy(); expected[:, :, NIGHT] = np.maximum(expected[:, :, NIGHT], 0)
    base_error = float(np.nanmax(abs(expected-base)))
    assert base_error == 0
    cap = a['capacity_kwp'][:, np.maximum(np.arange(len(a['date']))-1, 0)]
    records=[]
    for fee in ['tou', 'flat']:
        for method in ['net_mse_night', 'net_calibrated24_night']:
            name=f'{method}_{fee}_seed11_expanded'
            z=np.load(REV/f'dispatch/{name}.npz')
            mask=z['valid_gcgg_day']; signal=z['forecast_net_kw']
            night=signal[:, :, NIGHT][mask]
            rec={'method':method, 'tariff':fee, 'test_household_days':int(mask.sum()),
                 'test_night_values':int(night.size),
                 'test_negative_night_fraction':float(np.mean(night<0)),
                 'test_minimum_night_net_kw':float(night.min())}
            assert np.all(night>=0)
            if method=='net_calibrated24_night':
                report=json.loads((REV/f'results/{method}_{fee}_seed11.json').read_text())
                old=json.loads((REV/f'results/net_calibrated24_{fee}_seed11.json').read_text())
                assert report['training_pairs']==old['training_pairs']
                assert report['validation_pairs']==old['validation_pairs']
                assert report['night_nonnegative_net_projection_in_calibration_training_validation_inference']
                feats=design(a['date'], method)
                pred=raw+cap[:, :, None]*np.einsum('k,dkt->dt',report['coefficients'],feats)[None,:,:]
                pred[:,:,NIGHT]=np.maximum(pred[:,:,NIGHT],0)
                stored=np.load(REV/f'forecasts/{method}_{fee}_seed11.npz')['prediction_kw']
                err=float(np.nanmax(abs(pred.astype(np.float32)-stored)))
                assert err==0
                for split, date_limit in [('training_pairs','2012-01-01'),('validation_pairs','2012-07-01')]:
                    pairs=np.asarray(report[split]); dates=a['date'][pairs[:,1]]
                    assert np.all(dates<date_limit)
                    assert np.all(stored[pairs[:,0],pairs[:,1]][:,NIGHT]>=0)
                rec.update({'training_pairs':len(report['training_pairs']),
                            'validation_pairs':len(report['validation_pairs']),
                            'same_pairs_as_signed_calibrator':True,
                            'selected_alpha':report['selected']['alpha'],
                            'validation_cost':report['selected']['validation_operating_excluding_cl'],
                            'formula_reconstruction_max_error_kw':err,
                            'training_resources':report['training_resources']})
            records.append(rec)
    out={'status':'PASS','base_projection_exact_error_kw':base_error,
         'chronology':'Support correction was introduced after this revision diagnosed negative night net values. Original signed results are retained as exploratory comparisons. The correction is not an independent confirmation.',
         'scope':'The already-fitted supervised direct-net MLP weights remain unchanged. Its fixed night projection is applied at deployment and to the base signal used during calibration. The 24 offsets are fitted and validation-selected with projection inside the calibration training, validation and inference paths. This does not claim that foundational supervised training used the projection.',
         'projection':'max(net,0), fixed slots0:8 and42:48; daytime remains signed',
         'records':records, **runtime_info(start)}
    write_json(REV/'results/direct_net_night_support_audit.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
