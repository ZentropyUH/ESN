from abc import ABC, abstractmethod
import keras
import tensorflow as tf
from typing import List, Tuple, Union


@keras.saving.register_keras_serializable(package="MyLayers", name="BaseCell")
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
        self.state_size = units
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
        return [tf.zeros((batch_size, self.state_size), dtype=self.compute_dtype)]

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
            }
        )
        return config


@keras.saving.register_keras_serializable(package="MyLayers", name="BaseReservoir")
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

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        """
        Forward pass of the reservoir.

        Parameters
        ----------
        inputs : Union[tf.Tensor, List[tf.Tensor]]
            The feedback and input sequences for the reservoir. They both have shape (batch_size, timesteps, features). The input sequence can be None, in which case a zero tensor will be used.
        training : bool, optional
            Whether the call is in training mode, by default False

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

            total_seq = tf.concat([input_seq, feedback_seq], axis=-1)

        else:
            total_seq = inputs

        # Concatenated input and feedback sequences. Cell expects (batch_size, timesteps, feedback_dim + input_dim)
        return super().call(total_seq)

    def compute_output_shape(self, input_shape) -> Union[List[int], Tuple[int]]:
        """
        Computes the output shape of the reservoir.
        Since the RNN always returns the full sequence, the output shape is:
            (batch_size, timesteps, units)
        """
        if isinstance(input_shape, (list)):
            # Handle case where input is [feedback_seq, input_seq]
            feedback_shape = input_shape[0]  # feedback sequence shape
            batch_size, timesteps = feedback_shape[:2]
        else:
            # Single input case (feedback only)
            batch_size, timesteps = input_shape[:2]

        return (batch_size, timesteps, self.units)

    def get_config(self):
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
