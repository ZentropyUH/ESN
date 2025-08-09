import tensorflow as tf
from typing import List, Optional, Tuple, Union

from keras_reservoir_computing.utils.tensorflow import create_tf_rng
from keras_reservoir_computing.layers.reservoirs.cells import BaseCell


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
            trainable=False,
            **kwargs,
        )
        self.units = cell.units
        self.feedback_dim = cell.feedback_dim
        self.input_dim = cell.input_dim

        # State collection flag and buffer
        self._collect_states = False
        self._full_states = None

    def enable_state_collection(self, flag: bool = True) -> None:
        """Enable or disable collection of full state trajectories."""
        self._collect_states = flag
        if not flag:
            self._full_states = None

    def get_full_states(self) -> Optional[tf.Tensor]:
        """Return collected full states [batch, timesteps, units], or None if disabled."""
        return self._full_states

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

    def set_random_states(self, dist: str = "uniform", seed: Optional[int] = None) -> None:
        """
        Set the states of the reservoir to random values.

        Parameters
        ----------
        dist : str, optional
            The distribution to sample from. Can be "uniform" or "normal".
        seed : int, optional
            The seed for the random number generator.
        """

        rng = create_tf_rng(seed)

        if dist not in {"uniform", "normal"}:
            raise ValueError(
                f"Invalid distribution: {dist}. Should be 'uniform' or 'normal'."
            )

        for i in range(
            len(self.states)
        ):  # Ensures TensorFlow properly tracks assignment
            if dist == "uniform":
                self.states[i].assign(
                    rng.uniform(self.states[i].shape, -1.0, 1.0)
                )
            else:  # "normal"
                self.states[i].assign(rng.normal(self.states[i].shape))

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

            fb_feats = shape_fb[-1]

            if fb_feats != self.feedback_dim:
                raise ValueError(
                    f"Feedback sequence has {fb_feats} features, expected {self.feedback_dim}"
                )

            in_feats = shape_in[-1]

            if in_feats is not None and in_feats != self.input_dim:
                raise ValueError(
                    f"Input sequence has {in_feats} features, expected {self.input_dim}"
                )

            # If in_feats is unknown at build time, fall back to configured input_dim
            combined_features = fb_feats + (in_feats if in_feats is not None else self.input_dim)
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
            - input_sequence: shape (batch_size, timesteps, input_dim)


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

            # Concatenate feedback and input sequences along the last dimension. Cell expects (batch_size, timesteps, feedback_dim + input_dim). Will split internally
            total_seq = tf.concat([feedback_seq, input_seq], axis=-1)

        else:
            total_seq = inputs

        # Concatenated input and feedback sequences. Cell expects (batch_size, timesteps, feedback_dim + input_dim)
        outputs = super().call(total_seq)

        if self._collect_states:
            self._full_states = outputs  # already [batch, timesteps, units]
        else:
            self._full_states = None

        return outputs

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

    # Intentionally non-trainable (weights in the wrapped cell are also non-trainable)

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

    @classmethod
    def from_config(cls, config) -> "BaseReservoir":
        """
        Create an ``BaseReservoir`` from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration produced by ``get_config()``.

        Returns
        -------
        BaseReservoir
            A new layer instance.
        """
        return cls(**config)
