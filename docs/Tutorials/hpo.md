# Tutorial: Hyper-parameter optimization (HPO)

KRC ships a thin wrapper around Optuna to search over reservoir and readout hyper-parameters.

See the complete example in {doc}`../hyperparameter_optimization`.

## Typical study

```python
from optuna.trial import Trial
from keras_reservoir_computing.hpo import run_hpo

# Define model_creator, search_space(trial), and data_loader()
study = run_hpo(
    model_creator=model_creator,
    search_space=search_space,
    data_loader=data_loader,
    n_trials=50,
    loss="efh",
    loss_params={"metric": "rmse", "threshold": 0.2, "softness": 0.02},
    study_name="esn_hpo",
    storage="sqlite:///esn_hpo.db",
)
print("Best:", study.best_params)
```

- Prefer `leak_rate` around 0.3–1.0; set `spectral_radius` ≥ 1−leak.
- Tune `alpha` for `RidgeReadout` on a log scale.
