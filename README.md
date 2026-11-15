# Radar step counting and walking distance estimation: data and evaluation code

Data and scripts for the leakage-aware, subject-independent evaluation of step counting
and walking distance estimation from the tracker output of a TI IWR6843AOP mmWave radar.

## Contents

- `Dataset_All/` : 122 walking trials from three subjects (`S1`, `S2`, `S3`).
  Folder names encode the scenario as `<distance>m_<body height in cm>_<steps>steps`.
  Each CSV holds one trial: `timestamp` (ms), tracked position (`xpos`, `ypos`, `zpos`, m),
  velocity (`xvel`, `yvel`, `zvel`, m/s) and acceleration. Rows of a spurious second target
  may be interleaved (no target identifier was logged).
- `config/AOP_p3.5_l2.cfg` : radar chirp, frame and tracker configuration.
- `evaluation/` : scripts, result files and figures.

## Reproducing the results

```
pip install -r requirements.txt
cd evaluation
python run_ml.py random ET,RF,KNN,HGB,Ridge,Mean h   # random 80/20 window split (leaky)
python run_ml.py shuf   ET,KNN random                # shuffled-label control
python run_ml.py trial  ET,KNN,HGB,Ridge,Mean h      # trial-grouped 5-fold
python run_ml.py loso   ET,KNN,HGB,Ridge,Mean h      # leave-one-subject-out
python run_ml.py trial  ET,KNN noh                   # same, without the height feature
python run_ml.py loso   ET,KNN noh
python run_ml.py random ET,KNN noh
python run_base.py                                   # trajectory and cadence baselines
python run_clean.py loso:noh loso:h trial:noh trial:h  # Extra Trees on the cleaned track
python extra.py KNN:23 HGB:23 Ridge:23 Mean:23 RF:23 ET:12 ET:35
python paper.py                                      # tables (paper_table.csv) and figures (Fig/)
```

`paper_table.csv` holds the cumulative per-trial errors and `final_window_level.csv`
the per-window MAE and R2 for every validation protocol.
