from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="krc", name="BaseCell")
class BaseCell(tf.keras.Layer, ABC):
    """
    Abstract base class for different types of reservoir cells.

    A reservoir cell is the fundamental computational unit of a reservoir network,
    similar to an RNN cell. It defines how the reservoir state is updated for
    each time step based on inputs and previous states.

    Parameters
    ----------
    units : int
        Number of units (neurons) in the reservoir cell.
    feedback_dim : int, optional
        Dimensionality of the feedback input. Default is 1.
    input_dim : int, optional
        Dimensionality of the external input. Default is 1.
    leak_rate : float, optional
        Leaking rate for the reservoir state update, controls the speed of
        dynamics (0 = no update, 1 = complete update). Default is 1.0.
    state_sizes : List[int], optional
        List of state sizes for each state tensor. If None, a single state
        with size `units` is used. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the parent Layer class.

    Attributes
    ----------
    units : int
        Number of units in the reservoir cell.
    feedback_dim : int
        Dimensionality of the feedback input.
    input_dim : int
        Dimensionality of the external input.
    state_size : List[int]
        List of sizes for each state tensor.
    leak_rate : float
        Leaking rate for the reservoir state update.

    Notes
    -----
    - This is an abstract base class that must be subclassed
    - Subclasses must implement `build()` and `call()` methods
    - The state_size attribute can be a list of integers if the cell has multiple states

    Examples
    --------
    Subclassing example:

    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.reservoirs.base import BaseCell
    >>>
    >>> class SimpleReservoirCell(BaseCell):
    ...     def build(self, input_shape):
    ...         self.kernel = self.add_weight(
    ...             shape=(self.feedback_dim, self.units),
    ...             initializer="glorot_uniform",
    ...             name="kernel"
    ...         )
    ...         super().build(input_shape)
    ...
    ...     def call(self, inputs, states, training=False):
    ...         prev_state = states[0]
    ...         output = tf.matmul(inputs, self.kernel)
    ...         new_state = (1 - self.leak_rate) * prev_state + self.leak_rate * output
    ...         return new_state, [new_state]
    """

    def __init__(
        self,
        units: int,
        feedback_dim: int = 1,
        input_dim: int = 1,
        leak_rate: float = 1.0,
        state_sizes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the BaseCell.

        Parameters
        ----------
        units : int
            Number of units (neurons) in the reservoir cell.
        feedback_dim : int, optional
            Dimensionality of the feedback input. Default is 1.
        input_dim : int, optional
            Dimensionality of the external input. Default is 1.
        leak_rate : float, optional
            Leaking rate for the reservoir state update (0 = no update, 1 = complete update).
            Default is 1.0.
        state_sizes : List[int], optional
            List of state sizes for each state tensor. If None, a single state with
            size `units` is used. Default is None.
        **kwargs : dict
            Additional keyword arguments passed to the parent Layer class.

        Examples
        --------
        >>> from keras_reservoir_computing.layers.reservoirs.cells import ESNCell
        >>> # Create a basic ESN cell with 100 units
        >>> cell = ESNCell(units=100)
        >>> # Create a cell with custom parameters
        >>> cell = ESNCell(
        ...     units=150,
        ...     feedback_dim=10,
        ...     input_dim=5,
        ...     leak_rate=0.3,
        ...     spectral_radius=0.9
        ... )
        """
        super().__init__(**kwargs)
        self.units = units
        self.feedback_dim = feedback_dim
        self.input_dim = input_dim
        self.state_size = state_sizes if state_sizes is not None else [units]
        self.leak_rate = leak_rate

    @abstractmethod
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Create the weights of the reservoir cell.
        In any custom build always call `super().build(input_shape)` in the end.

        Parameters
        ----------
        input_shape : tf.TensorShape
            Shape of the inputs.
        """
        pass

    @abstractmethod
    def call(
        self, inputs: tf.Tensor, states: List[tf.Tensor], training: bool = False
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Forward pass of one time step of the reservoir cell.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor for the current time step.
        states : List[tf.Tensor]
            Previous state(s) of the reservoir.
        training : bool, optional
            Whether the call is in training mode, by default False

        Returns
        -------
        tf.Tensor
            The new output state of the reservoir cell.
        List[tf.Tensor]
            A list containing the new state(s).
        """
        pass

    def get_initial_state(self, batch_size: int = None) -> List[tf.Tensor]:
        """
        Generate initial state tensors for the reservoir.

        Creates initial states for the reservoir with random values
        drawn from a uniform distribution in the range [-1, 1].

        Parameters
        ----------
        batch_size : int, optional
            The batch size for the generated states. Must be provided
            to create properly shaped state tensors.

        Returns
        -------
        List[tf.Tensor]
            A list of tensors representing the initial state.
            Each tensor has shape (batch_size, state_size_i).

        Examples
        --------
        >>> from keras_reservoir_computing.layers.reservoirs.cells import ESNCell
        >>> cell = ESNCell(units=10)
        >>> states = cell.get_initial_state(batch_size=4)
        >>> [s.shape for s in states]
        [(4, 10)]

        Notes
        -----
        This method is typically called by the RNN layer that wraps the cell
        to initialize the states at the beginning of a sequence.
        """
        return [
            tf.random.uniform(
                (batch_size, size), minval=-1.0, maxval=1.0, dtype=self.compute_dtype
            )
            for size in self.state_size
        ]

    def get_config(self) -> dict:
        """
        Return the configuration dictionary of the BaseCell.

        Returns
        -------
        dict
            Dictionary containing all configuration parameters needed
            to reconstruct the cell (units, feedback_dim, input_dim,
            leak_rate, state_sizes).

        Examples
        --------
        >>> from keras_reservoir_computing.layers.reservoirs.cells import ESNCell
        >>> cell = ESNCell(units=100, leak_rate=0.5)
        >>> config = cell.get_config()
        >>> # Create a new cell with the same configuration
        >>> new_cell = ESNCell.from_config(config)
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "feedback_dim": self.feedback_dim,
                "input_dim": self.input_dim,
                "leak_rate": self.leak_rate,
                "state_sizes": self.state_size,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="krc", name="BaseReservoir")
class BaseReservoir(tf.keras.layers.RNN):
    """
    Abstract base class for reservoir layers.

    A reservoir layer wraps a reservoir cell (BaseCell) within TensorFlow's RNN
    framework to process sequence data. This base class ensures that all reservoir
    implementations are stateful and return the full sequence of outputs.

    Parameters
    ----------
    cell : BaseCell
        The reservoir cell to use in this layer. Must be a subclass of BaseCell.
    **kwargs : dict
        Additional keyword arguments passed to the parent RNN class.

    Attributes
    ----------
    cell : BaseCell
        The reservoir cell used in the layer.
    units : int
        Number of units (neurons) in the reservoir.
    feedback_dim : int
        Dimensionality of the feedback input.
    input_dim : int
        Dimensionality of the external input.

    Notes
    -----
    - All reservoirs are configured as stateful by default
    - All reservoirs return sequences by default (return_sequences=True)
    - State management methods (get_states, set_states, reset_states) allow
      explicit control of the reservoir's internal state
    - Use reset_states() to set all states to zero
    - Use set_random_states() to randomize all states

    Examples
    --------
    >>> import tensorflow as tf
    >>> from keras_reservoir_computing.layers.reservoirs.cells import ESNCell
    >>> from keras_reservoir_computing.layers.reservoirs.reservoirs import ESNReservoir
    >>>
    >>> # Create a simple ESN
    >>> inputs = tf.keras.Input(shape=(None, 5))  # Sequential input with 5 features
    >>> cell = ESNCell(units=100, spectral_radius=0.9, leak_rate=0.3)
    >>> reservoir = ESNReservoir(cell=cell)(inputs)
    >>> model = tf.keras.Model(inputs, reservoir)
    >>>
    >>> # Examine the output shape
    >>> print(f"Output shape: {model.output_shape}")
    Output shape: (None, None, 100)  # (batch_size, timesteps, units)
    >>>
    >>> # Create input data and run the model
    >>> x = tf.random.normal((32, 10, 5))  # (batch_size=32, timesteps=10, features=5)
    >>> output = model(x)
    >>> print(f"Output tensor shape: {output.shape}")
    Output tensor shape: (32, 10, 100)
    """

    def __init__(
        self,
        cell: BaseCell,
        **kwargs,
    ) -> None:
        """
        Initialize the BaseReservoir.

        Parameters
        ----------
        cell : BaseCell
            The reservoir cell to use in the reservoir.
        **kwargs : dict
            Additional keyword arguments for the RNN base class.
        """
        super().__init__(
            cell=cell,
            stateful=True,
            return_sequences=True,
            return_state=False,
            **kwargs,
        )
        self.units = cell.units
        self.feedback_dim = cell.feedback_dim
        self.input_dim = cell.input_dim

    def get_states(self) -> List[tf.Tensor]:
        """
        Return the states of the reservoir.

        Returns
        -------
        List[tf.Tensor]
            List containing the states of the reservoir.
        """
        return [tf.identity(state) for state in self.states]

    def set_states(self, states: List[tf.Tensor]) -> None:
        """
        Set the states of the reservoir.

        Parameters
        ----------
        states : List[tf.Tensor]
            List containing the states of the reservoir.
        """
        # validate number of states and shapes
        if len(states) != len(self.states):
            raise ValueError(
                "Number of states does not match the number of reservoir states."
            )
        for state, new_state in zip(self.states, states):
            if state.shape != new_state.shape:
                raise ValueError(
                    f"State shape mismatch. Existing: {state.shape}, new: {new_state.shape}."
                )
        for s, new_s in zip(self.states, states):
            s.assign(new_s)

    @tf.function
    def set_random_states(self, dist: str = "uniform") -> None:
        """
        Set the states of the reservoir to random values.

        Parameters
        ----------
        dist : str, optional
            The distribution to sample from. Can be "uniform" or "normal".
        """
        if dist not in {"uniform", "normal"}:
            raise ValueError(
                f"Invalid distribution: {dist}. Should be 'uniform' or 'normal'."
            )

        for i in range(
            len(self.states)
        ):  # Ensures TensorFlow properly tracks assignment
            if dist == "uniform":
                self.states[i].assign(
                    tf.random.uniform(self.states[i].shape, -1.0, 1.0)
                )
            else:  # "normal"
                self.states[i].assign(tf.random.normal(self.states[i].shape))

    def build(self, input_shape) -> None:
        """
        Keras will pass the 'input_shape' that this RNN layer sees at build-time.
        It might be:
         - a single shape (batch_size, timesteps, features)
         - a list of two shapes [ (batch_size, timesteps, fb_feats),
                                  (batch_size, timesteps, in_feats) ]
           or something partially defined (None for batch or timesteps).

        We must figure out the "combined" shape that our cell expects:
            (batch_size, timesteps, feedback_dim + input_dim)
        Then call super().build(...) with that shape so it can build the cell.
        """

        if all(isinstance(shape, (list, tuple)) for shape in input_shape):
            shape_fb, shape_in = map(tuple, input_shape)
            # shape_fb = (batch_size, timesteps, fb_feats)
            # shape_in = (batch_size, timesteps, in_feats)
            # Some or all dims might be None at this stage.

            # If 'input_dim' is supposed to be 0 or if the input is genuinely None,
            # we won't know in advance. Typically you want shape_in[-1] == self.input_dim,
            # shape_fb[-1] == self.feedback_dim.
            # We'll define an aggregated 'features' dimension for the cell.

            fb_feats = shape_fb[-1]

            if fb_feats != self.feedback_dim:
                raise ValueError(
                    f"Feedback sequence has {fb_feats} features, expected {self.feedback_dim}"
                )

            in_feats = shape_in[-1] if shape_in is not None else self.input_dim

            if in_feats is not None and in_feats != self.input_dim:
                raise ValueError(
                    f"Input sequence has {in_feats} features, expected {self.input_dim}"
                )

            combined_features = fb_feats + in_feats
            # Now define a synthetic shape to pass to super().build(...).
            # We only really need (batch_size, timesteps, combined_features).
            # The batch_size or timesteps might be None, but that’s fine.
            shape_total = (shape_fb[0], shape_fb[1], combined_features)

        else:
            # Single input shape. We assume it’s the feedback sequence only.
            shape_total = tuple(input_shape)
            if shape_total[-1] != self.feedback_dim:
                raise ValueError(
                    f"Feedback sequence has {shape_total[-1]} features, expected {self.feedback_dim}"
                )

        # Now let the RNN (and thus the cell) build weights for that total dimension.
        super().build(shape_total)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        Forward pass of the reservoir.

        Parameters
        ----------
        inputs : Union[tf.Tensor, List[tf.Tensor]]
            The input to the reservoir. If a single tensor is provided, it is assumed to be only the feedback sequence.
            If a list of two tensors is provided, it must be [feedback_sequence, input_sequence], where:
            - feedback_sequence: shape (batch_size, timesteps, feedback_dim)
            - input_sequence: shape (batch_size, timesteps, input_dim) (can be None, replaced by zeros).


        Returns
        -------
        tf.Tensor
            The output sequence of the reservoir.
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(
                    "Input must be a list of two tensors: [feedback_sequence, input_sequence]."
                )

            feedback_seq, input_seq = inputs

            if input_seq is None:
                batch_size = tf.shape(feedback_seq)[0]
                timesteps = tf.shape(feedback_seq)[1]

                # Create a zero tensor for the input sequence
                zeros_shape = (batch_size, timesteps, self.input_dim)
                input_seq = tf.zeros(zeros_shape)

            total_seq = tf.concat([feedback_seq, input_seq], axis=-1)

        else:
            total_seq = inputs

        # Concatenated input and feedback sequences. Cell expects (batch_size, timesteps, feedback_dim + input_dim)
        return super().call(total_seq)

    def compute_output_shape(self, input_shape) -> Tuple[int, ...]:
        """
        Computes the output shape of the reservoir.
        Since the RNN always returns the full sequence, the output shape is:
            (batch_size, timesteps, units)
        """
        if all(isinstance(shape, (list, tuple)) for shape in input_shape):
            input_shape = input_shape[0]  # feedback sequence shape
        batch_size, timesteps = input_shape[:2]

        return (batch_size, timesteps, self.units)

    def get_config(self) -> dict:
        """
        Ensure that hardcoded parameters (return_sequences, stateful, etc.)
        are removed from the saved configuration to avoid redundancy and deserialization issues.
        """
        base_config = super().get_config()

        # Remove parameters that are hardcoded in __init__()
        for param in [
            "return_sequences",
            "return_state",
            "stateful",
        ]:
            base_config.pop(param, None)  # Remove only if present

        return base_config
