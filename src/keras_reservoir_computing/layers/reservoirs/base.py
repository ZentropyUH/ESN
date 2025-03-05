from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="krc", name="BaseCell")
class BaseCell(keras.Layer, ABC):
    """
    Abstract base class for different types of reservoir cells.
    Each reservoir cell represents the one-step computation unit
    of a reservoir (akin to an RNN cell).

    Parameters
    ----------
    units : int
        Number of units in the reservoir cell.
    feedback_dim : int, optional
        Dimensionality of the feedback input, by default 1.
    input_dim : int, optional
        Dimensionality of the input, by default 1.
    leak_rate : float, optional
        Leak rate of the reservoir cell, by default 1.0.
    activation : Optional[Union[str, Callable]], optional
        Activation function to use, by default "tanh".
    **kwargs : dict
        Additional keyword arguments for the Layer base class.

    Attributes
    ----------
    units : int
        Number of units in the reservoir cell.
    state_size : int
        Size of the state (same as `units`).
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
        Initialize the BaseReservoirCell.

        Parameters
        ----------
        units : int
            Number of units in the reservoir cell.
        **kwargs : dict
            Additional keyword arguments for the Layer base class.
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
        return [
            tf.random.uniform(
                (batch_size, size), minval=-1.0, maxval=1.0, dtype=self.compute_dtype
            )
            for size in self.state_size
        ]

    def get_config(self) -> dict:
        """
        Return the configuration of the BaseReservoirCell.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "feedback_dim": self.feedback_dim,
                "input_dim": self.input_dim,
                "leak_rate": self.leak_rate,
                "state_sizes": self.state_size
            }
        )
        return config


@keras.saving.register_keras_serializable(package="krc", name="BaseReservoir")
class BaseReservoir(keras.layers.RNN):
    """
    Abstract base class for different types of reservoirs.
    Each reservoir is a recurrent neural network that wraps a reservoir cell in a RNN.
    All reservoirs will be stateful and return the full sequence of outputs.

    Parameters
    ----------
    cell : BaseCell
        The reservoir cell to use in the reservoir.
    **kwargs : dict
        Additional keyword arguments for the RNN base class.

    Attributes
    ----------
    cell : BaseCell
        The reservoir cell to use in the reservoir.
    return_sequences : bool
        Whether to return the full sequence of outputs.
    return_state : bool
        Whether to return the last state of the reservoir.
    state_size : int
        Size of the reservoir state.

    Methods
    -------
    get_states(self) -> List[tf.Tensor]
        Return the states of the reservoir.
    set_states(self, states: List[tf.Tensor]) -> None
        Set the states of the reservoir.
    _harvest(self, sequence: tf.Tensor) -> List[tf.Tensor]]
        Harvest the reservoir states from the input sequence.
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
        return self.states

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
                    "Input must be a list of two tensors: [input_sequence, feedback_sequence]."
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
