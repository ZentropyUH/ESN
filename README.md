# ESN
An Echo State Network implementation integrated with a general scheme to use broad range of other reservoirs.

## Installation

### For those that use Anaconda

Create a conda environment with the `environment.yml` file provided in the repository.

```bash
conda env create -f environment.yml
```

After creating the environment, you should activate it.

```bash
conda activate krc
```

### For those that use virtualenv

Create a virtual environment with the `requirements.txt` file provided in the repository.

```bash
virtualenv krc
source krc/bin/activate
```

### For those who prefer to live on the edge

Skip the environment creation and install the dependencies manually. Let Troy be with you.


### Install the package

Then you can install the package using the `setup.py` file.

```bash
python setup.py
```

Alternatively, you can install the package using pip, standing in the root directory of the repository.

```bash
pip install .
```


From then on you can import the package in your code.

```python
import keras_reservoir_computing as krc

krc.utils # A module with utility functions
krc.initializers # A module with custom initializers
krc.layers # A module with custom layers
krc.reservoirs # A module with custom reservoirs
krc.models # A module with custom models
```

## General description

The main classes are `ReservoirComputer` which is a general class to use any reservoir, and `ReservoirEnsemble`, which is a class to use several reservoir computers integrated in an ensemble mode. 

The reservoirs are implemented in the `reservoirs` module.


`BaseReservoir` and `BaseReservoirCell` are the abstract classes for the reservoirs and reservoir cells, respectively. 

The `EchoStateNetwork` class is a subclass of `BaseReservoir`. The `EchoStateNetwork` class is a simple implementation of an ESN with its `ESNCell` class being the reservoir cell.


### Initializers

The custom initializers allow for a more flexible way to initialize the reservoir. 

The `InputMatrix` class is used to initialize the feedback weights. The `WattsStrogatzNX` class is used to initialize the reservoir weights. 

The `WattsStrogatzNX` class is a subclass of `keras.initializers.Initializer` and it is used to initialize the reservoir weights using the Watts-Strogatz model. The `WattsStrogatzNX` class uses the `networkx` library to generate the Watts-Strogatz graph.


## Usage


### ReservoirComputer

To use the `ReservoirComputer` class we have to define the reservoir we want to use. Currently available reservoirs are:

- `EchoStateNetwork` (With feedback only)

#### Code example

##### Define the hyperparameters if needed. Optional but recommended.

```python
input_scaling = 0.1 # This is the scaling of the input weights. Used to set the range of random values for the input weights
degree = 10 # This is the degree of the Watts-Strogatz graph
spectral_radius = 0.9 # Specially important for ESN to achieve the echo state property
rewiring = 0.9 # This is the probability of rewiring the Watts-Strogatz graph
units = 300 # This is the number of neurons in the reservoir. It saturates depending on the input size.
leak_rate = 0.6 # This regulates the speed of the reservoir dynamics
regularization = 1e-6 # Regularization for the readout layer
```

##### Instantiate initializers for the reservoir cell. This is optional, but the default values are not recommended.

```python
feedback_init = InputMatrix(sigma=input_scaling, ones=False, seed=seed)

feedback_bias_init = keras.initializers.random_uniform(seed=seed, minval=-input_scaling, maxval=input_scaling)

kernel_init = WattsStrogatzNX(
    degree=degree,
    spectral_radius=spectral_radius,
    rewiring_p=rewiring,
    # sigma=0.5,
    ones=True,
    seed=seed,
)
```


##### Load the data using the load_data function from utils.py
```python
data_file = (
    "./src/systems/data/Lorenz/Lorenz_dt0.02_steps150000_t-end3000.0_seed667850.csv"
    # "./src/systems/data/Lorenz/Lorenz_dt0.02_steps150000_t-end3000.0_seed28295.csv"
)
train_length = 20000 # Or any other value

transient_data, train_data, train_target, ftransient, val_data, val_target = load_data(
    data_file, train_length=train_length, transient=1000, normalize=True
)

features = transient_data.shape[-1]

# This will yield a tuple of np.ndarrays with shapes:
# (1, transient, 3)
# (1, train_length, 3)
# (1, train_length, 3)
# (1, transient, 3)
# (1, rest_of_timesteps, 3)
# (1, rest_of_timesteps, 3)
```

##### Instantiate the reservoir cell and the reservoir computer
```python
esn_cell = ESNCell(
    units=units,
    leak_rate=leak_rate,
    noise_level=0.0,
    input_initializer=feedback_init,
    input_bias_initializer=feedback_bias_init,
    kernel_initializer=kernel_init)

reservoir = EchoStateNetwork(
    reservoir_cell=esn_cell
)

readout_layer = keras.layers.Dense(
    features, activation="linear", name="readout", trainable=False
)

model = ReservoirComputer(reservoir=reservoir, readout=readout_layer, seed=seed)
```

##### Train the reservoir computer

Since the training of ESN is a non-conventional training, we have a custom train method within the `ReservoirComputer` class. This will be used to train the readout layer instead of the conventional `model.fit` method.

```python
loss = model.train(
    (transient_data, train_data), train_target, regularization=regularization
)
```

##### Predict using the reservoir computer

The predictions are done using the custom method `forecast` from the `ReservoirComputer` class.

This method will return the forecast, the internal states of the reservoir, the cumulative error, and the number of steps that the error was above the threshold.

The `states` are returned as a 2D Tensor with shape `(1, n_timesteps, n_units)` only if internal_states is set to True.

The `cumulative` error is a 1D Tensor with shape `(n_timesteps,)` and it is the sum of the errors for each timestep.

The `threshold_steps` is the number of steps before the error was above the threshold. Only returned if the error_threshold is set.

The `forecast` is a 2D Tensor with shape `(1, forecast_length, n_features)`.


```python
forecast_length = 1000
forecast, states, cumulative_error, threshold_steps = model.forecast(
    forecast_length=forecast_length,
    forecast_transient_data=ftransient,
    val_data=val_data,
    val_target=val_target,
    error_threshold=0.5,
    internal_states=True,
)
```


### ReservoirEnsemble

TODO
