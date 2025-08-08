# Tutorial: Custom reservoirs and initializers

Create reservoirs with tailored topology using graph-based initializers.

```python
import tensorflow as tf
from keras_reservoir_computing.initializers.recurrent_initializers.graph_initializers import (
    WattsStrogatzGraphInitializer,
    BarabasiAlbertGraphInitializer,
    ErdosRenyiGraphInitializer,
)
from keras_reservoir_computing.layers import ESNReservoir
from keras_reservoir_computing.layers.readouts import RidgeReadout
```

## Watts–Strogatz reservoir

```python
ws_kernel = WattsStrogatzGraphInitializer(
    k=8, p=0.2, spectral_radius=0.95, directed=True, self_loops=True, seed=7
)
reservoir = ESNReservoir(
    units=500,
    feedback_dim=1,
    input_dim=0,
    leak_rate=0.7,
    kernel_initializer=ws_kernel,
)

inp = tf.keras.layers.Input(shape=(None, 1), batch_size=1)
states = reservoir(inp)
out = RidgeReadout(units=1, alpha=1e-2, name="readout")(tf.keras.layers.Concatenate()([inp, states]))
model = tf.keras.Model(inp, out)
```

## Scale-free reservoir

```python
ba_kernel = BarabasiAlbertGraphInitializer(m=3, spectral_radius=0.9, directed=True)
```

## Random reservoir

```python
er_kernel = ErdosRenyiGraphInitializer(p=0.1, spectral_radius=0.9, directed=True)
```

Swap `kernel_initializer` to experiment with different dynamics.