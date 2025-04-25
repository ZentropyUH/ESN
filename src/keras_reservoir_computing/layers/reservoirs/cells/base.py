import tensorflow as tf
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


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
    ...     def call(self, inputs, states):
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

