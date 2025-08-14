# Design rationale and architecture

KRC exists because previous attempts to shoehorn reservoir computing into high-level deep learning libraries often conflated responsibilities: they trained reservoirs, obscured data flow, or required awkward custom loops. We designed KRC to be idiomatic Keras while staying faithful to reservoir computing principles.

## Design principles

- Keep reservoirs fixed: reservoirs are initialized richly and then frozen.
- Make readouts explicit: readouts are trainable layers with a simple `.fit` API.
- Lean into the Keras graph: use functional models and reuse subgraphs for training/forecasting.
- Zero-ambiguity forecasting: helpers that make warm-up and auto-regressive rollout explicit and inspectable.

## Custom training

`ReservoirTrainer` avoids touching reservoir weights. It discovers each `ReadOut` in topological order, constructs a submodel that surfaces that readout’s input tensor, warms the model, and then calls `ReadOut.fit`. This makes pipelines predictable and memory efficient.

## Custom forecasting

`warmup_forecast` composes two phases:

1. Teacher-forced warm-up through the full model to settle states.
2. Auto-regressive rollout that feeds predictions back as inputs. Optionally tracks every `BaseReservoir` layer’s state via TensorArrays for analysis.

This separation improves reproducibility and integrates cleanly with multi-input models (exogenous variables).

## Why Keras?

- Composability: Functional API makes it trivial to wire reservoirs into large models.
- Tooling: Serialization, saving/loading, device placement, mixed precision.
- Ecosystem: Works alongside the broader TensorFlow toolchain, datasets, and serving.