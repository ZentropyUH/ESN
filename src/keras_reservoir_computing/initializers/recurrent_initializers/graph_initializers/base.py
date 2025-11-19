from typing import List, Optional, Tuple, Union

import tensorflow as tf

from keras_reservoir_computing.initializers.helpers import (
    create_rng,
    spectral_radius_hybrid,
)


@tf.keras.utils.register_keras_serializable(package="krc", name="GraphInitializerBase")
class GraphInitializerBase(tf.keras.Initializer):
    """
    Base class for initializers generating adjacency matrices for graph-based models.

    This initializer constructs adjacency matrices based on a specified graph generation
    function. It allows for optional spectral radius control to adjust the eigenvalues
    of the generated matrix.

    Parameters
    ----------
    spectral_radius : float or None, optional
        Desired spectral radius of the adjacency matrix. If None, no rescaling is applied.
    seed : int or None, optional
        Random seed for reproducibility.

    Methods
    -------
    __call__(shape, dtype=None)
        Generates an adjacency matrix with the specified shape.
    _generate_adjacency_matrix(n, *args, **kwargs)
        Abstract method for generating a graph adjacency matrix.
    get_config()
        Returns a dictionary of the initializer's configuration.

    Returns
    -------
    Tensor
        A 2D adjacency matrix representing the generated graph.

    Notes
    -----
    This is an abstract base class and must be subclassed with a specific graph
    generation function implemented in `_generate_adjacency_matrix`.
    """

    def __init__(
        self,
        spectral_radius: Optional[float] = None,
        seed: Union[int, tf.random.Generator, None] = None,
    ) -> None:
        if spectral_radius is not None and spectral_radius < 0:
            raise ValueError("The spectral radius should be non-negative.")

        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = create_rng(seed)
        super().__init__()

    def __call__(
        self,
        shape: Union[int, Tuple[int, int], List[int]],
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
        dims = tf.TensorShape(shape).as_list()  # -> list[int|None]

        if dims is None:
            raise ValueError("Rank of shape unknown at initialization time.")
        if len(dims) == 1:
            rows = int(dims[0])
        elif len(dims) == 2:
            rows, cols = map(int, dims)
        else:
            raise ValueError(f"Shape must be 1D or 2D, got {shape}")

        adj = self._generate_adjacency_matrix(rows)

        if self.spectral_radius is not None:
            sr = spectral_radius_hybrid(adj)
            adj = adj * self.spectral_radius / sr

        # Return the adjacency matrix as a 2D tensor
        return tf.convert_to_tensor(adj, dtype=dtype)

    def _generate_adjacency_matrix(
        self,
        n: int,
        *args,
        **kwargs,
    ) -> tf.Tensor:
        """
        Abstract method for generating a graph adjacency matrix.

        This method should be implemented by subclasses to generate the adjacency matrix
        for a specific type of graph.

        Parameters
        ----------
        n : int
            The number of nodes in the graph.
        *args : tuple
            Additional positional arguments for the graph generation function.
        **kwargs : dict
            Additional keyword arguments for the graph generation function.

        Returns
        -------
        tf.Tensor
            A 2D adjacency matrix representing the generated graph.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("The adjacency matrix generation function is not implemented.")

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        base_config = super().get_config()

        config = {
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }  # seed and spectral_radius are handled here
        config.update(base_config)
        return config
