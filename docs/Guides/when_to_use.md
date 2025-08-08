# When should I use X vs Y?

Use this guide to pick the right building blocks.

## Models

- `classic_ESN`: Default choice. Concatenates input and reservoir states before the readout. Great baseline for forecasting tasks.
- `Ott_ESN`: Uses selective exponentiation to augment states (squaring even indices). Try when the dynamics benefit from simple polynomial features.
- `headless_ESN`: No readout. Use for analyzing reservoir dynamics or when you plan to stack custom downstream heads.
- `linear_ESN`: Same as headless but with linear activation in the reservoir. Useful for linear-system analysis.

## Reservoir initializers

- Random recurrent (`RandomRecurrentInitializer`): Simple, fast baseline. Control `density` and `spectral_radius`.
- Graph-based (Watts–Strogatz, Erdős–Rényi, Barabási–Albert, etc.): Use when network topology matters. Set `directed`, `self_loops`, and `spectral_radius`.
- Input initializers (Random, PseudoDiagonal, Chebyshev, Chessboard): Choose depending on scaling and structure you want at the inputs.

## Readouts

- `RidgeReadout`: Regularized linear regression. Prefer as default; tune `alpha`.
- `MoorePenroseReadout`: Pseudo-inverse solution. Fast but can be less stable; good for quick experiments.

## Training

- Use `ReservoirTrainer` when your model contains one or more `ReadOut` layers. It trains them in dependency order using intermediate submodels.
- Provide a short `warmup_data` slice to settle reservoir dynamics, then pass the full `input_data` to accumulate readout inputs.

## Forecasting

- `warmup_forecast`: Recommended. Teacher-forced warm-up + auto-regressive rollout; can return hidden state histories.
- Pure auto-regressive: Use the forecast factories when you already have an initial feedback vector and optionally exogenous inputs.