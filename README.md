# Decision-focused learning for residential battery control

Code and processed data for the manuscript

> When does decision-focused learning add value to residential battery control?
> Private costs and AC feeder consequences

Submitted to *Electrical Engineering* (Springer).

## What the study asks

A residential battery controller already optimises an announced tariff, so a learning system
that is trained against realised operating cost has to justify itself against two things a
practitioner can build quickly: a low-dimensional calibration of the existing forecast, and a
scenario controller that represents the residual uncertainty the deterministic signal ignores.
The study measures how much of a cost-trained controller's benefit survives against those
alternatives, and what the changed battery action does to voltages on the feeder that hosts it.

## Headline results

- Over 108,641 paired household-days the decision loss lowers operating cost by 1.131
  Australian cents per day against continued squared-error training.
- Three trainable offsets already recover 54.1% of that gain; a 20-scenario controller
  reaches a further 2.462 cents below decision-focused learning.
- Matched screening-band voltage exposure falls by 0.853 percentage points at a 1.050 pu
  source, by 0.00848 points at 1.025 pu, and the two alternative networks show no
  screening-band events at all.
- A deployable load substitution can reverse the private gain until both learners are
  retrained, and when it does, overvoltage exposure rises on both evaluated panels.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | Analysis, training, dispatch, network and figure code |
| `configs/` | Run configuration |
| `checkpoints/` | Trained model checkpoints used by the reported comparisons |
| `datasets/` | Processed Ausgrid panel, network definitions and split records |
| `results/` | Result JSON, CSV and figure source data |
| `manuscript/` | Structured manuscript sources and the generated text |
| `revision/models/` | Revision model definitions and weights |
| `revision/network_models/`, `revision/network_*` | Network models, diagnostics and mechanism outputs |
| `revision/results/`, `revision/research/` | Revision results, audits and evidence records |
| `THIRD_PARTY_NOTICES.txt` | Sources, licences and attribution |

## Data sources

- **Ausgrid Solar Home Electricity Data** (2010 to 2013), distributed through data.gov.au.
  The processed panel under `datasets/` is derived from it.
- **IEEE European Low Voltage Test Feeder**, published by the IEEE PES Distribution System
  Analysis Subcommittee.
- **SimBench** rural and semiurban benchmark systems, and **pandapower** for balanced replay.
- **StoreNet** Irish energy-community load profiles (Trivedi et al., Scientific Data 11:621, 2024)
  for the independent transfer check.

No raw third-party download is redistributed here beyond the processed derivatives listed above.

## Environment

Python 3.12 with NumPy, pandas, PyTorch, scikit-learn, joblib, Clarabel, matplotlib,
SciencePlots, OpenDSSDirect, SimBench and pandapower. The recorded build environment is in
`manuscript/` alongside the run records; exact versions are listed in the manuscript's
resource table and in the reproduction notes.

## Reproducing the reported numbers

1. `python scripts/analyze_dispatch.py` recomputes settlement, throughput charges and battery
   state from the stored dispatch actions and the processed panel, and checks the reported
   paired mean differences.
2. `python scripts/analyze_network.py` and `scripts/analyze_robustness.py` reproduce the
   feeder and sensitivity summaries.
3. `python scripts/build_article_evidence.py` rebuilds the table and figure source data used
   by the manuscript.
4. `python scripts/revision_figures.py` regenerates the four multi-panel figures.

Training is not repeated by these steps; they operate on the retained checkpoints and saved
dispatch. The manuscript states this boundary explicitly.

## Citation

See `CITATION.cff`. The archived snapshot is deposited on Zenodo; the DOI is recorded in the
manuscript's data availability statement and in the Zenodo badge of this repository's releases.

## Licence

Code is released under the MIT Licence. Processed data retain the licences of their sources,
which are listed in `THIRD_PARTY_NOTICES.txt`.
