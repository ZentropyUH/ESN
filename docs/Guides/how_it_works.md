# How does this library work?

KRC implements Reservoir Computing for Keras with a clear separation of concerns:

- Reservoir layers produce rich hidden dynamics but keep internal weights fixed after initialization.
- Readout layers are the only trainable components and map hidden states to outputs.
- Training is handled by a dedicated `ReservoirTrainer` that respects model topology and only fits readouts.
- Forecasting utilities turn trained models into auto-regressive forecasters with optional state tracking.

## Architecture at a glance

- `keras_reservoir_computing.layers.reservoirs` – Reservoirs and cells (e.g., `ESNReservoir`, `ESNCell`)
- `keras_reservoir_computing.layers.readouts` – Trainable readouts (e.g., `RidgeReadout`, `MoorePenroseReadout`)
- `keras_reservoir_computing.models` – High-level model builders (`classic_ESN`, `Ott_ESN`, `headless_ESN`, `linear_ESN`)
- `keras_reservoir_computing.training` – `ReservoirTrainer` for fitting readouts
- `keras_reservoir_computing.forecasting` – `warmup_forecast` and helpers
- `keras_reservoir_computing.initializers` – Input/recurrent initializers, including graph-based reservoirs
- `keras_reservoir_computing.hpo` – Hyper-parameter optimization helpers
- `keras_reservoir_computing.utils` – Data utilities, visualization, TensorFlow helpers

## Typical workflow

1. Build a model using `classic_ESN` or assemble with the Keras Functional API.
2. Initialize reservoir connectivity with one of our initializers (random, graph-based, etc.).
3. Fit readout layers with `ReservoirTrainer` using a warm-up sequence and training data.
4. Forecast auto-regressively via `warmup_forecast` or pure `forecast` factories.

## Data flow

- Feedback input (the to-be-predicted variable) always feeds the reservoir’s first input.
- Optional external inputs are concatenated to the feedback inside the reservoir or before it.
- Reservoir outputs are often concatenated with the original inputs before the readout, improving observability (see `classic_ESN`).

## State handling

Reservoir layers are subclasses of `BaseReservoir` and can expose their hidden states via `get_states()`. Forecast helpers can capture per-timestep state histories for analysis and plotting.