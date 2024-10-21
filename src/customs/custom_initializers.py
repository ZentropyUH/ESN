"""Custom Initializers."""

from typing import Dict, Union, List, Tuple

import networkx as nx
import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.sparse import linalg

import keras
from keras import Initializer


@keras.saving.register_keras_serializable(package="MyInitializers", name="InputMatrix")
class InputMatrix(Initializer):
    """
    Makes an initializer that generates an input matrix which connects the input to the reservoir.

    Every node in the network receives exactly one scalar input and
    every input is connected to N/D nodes. N being the number of nodes (columns)
    and D is the number of inputs (rows).
    Every non-zero element in the matrix is randomly generated from
    a uniform distribution in [-sigma,sigma]

    Args:
        sigma (float): Standard deviation of the uniform distribution.

    Returns:
        keras.initializers.Initializer: The initializer.

    #### Usage example:
    >>> w_init = InputMatrix(sigma=1)
    >>> w = tf.Variable(w_init((5,10)))
    >>> w is a 5x10 matrix with values in [-1, 1]
    """

    def __init__(
        self, sigma: float = 0.5, ones: bool = False, seed: Union[int, None] = None
    ) -> None:
        """Initialize the initializer."""
        assert sigma > 0, "sigma must be positive"

        self.sigma = sigma
        self.ones = ones
        self.seed = seed

        if seed is not None:
            self.tf_rng = tf.random.Generator.from_seed(seed)
        else:
            self.tf_rng = tf.random.Generator.from_non_deterministic_state()

    def __call__(
        self,
        shape: Union[int, Tuple[int, int], List[int]],
        dtype=tf.float32,
    ) -> tf.Tensor:
        """
        Generate the matrix.

        Args:
            shape (tuple): Shape of the matrix.
            dtype (tf.dtype): Data type of the matr**kwargs
        """
        if isinstance(shape, int):
            rows, cols = (shape, shape)
        elif isinstance(shape, (list, tuple)) and len(shape) == 2:
            rows, cols = tuple(shape)
        else:
            raise ValueError("Shape must be an integer or a tuple/list of 2 integers")

        dense_shape = (rows, cols)

        assert (
            rows <= cols
        ), "Reservoir nodes must be greater than or equal to number of features of the input."

        inputs_per_node = int(cols / rows)

        indexes_list = []
        counter = 0

        # Assign base connections
        for i in range(rows):
            for j in range(inputs_per_node):
                index = counter % cols
                indexes_list.append([i, index])
                counter += 1

        # Distribute remainder
        remainder = cols % rows
        for i in range(remainder):
            index = counter % cols
            indexes_list.append([i, index])
            counter += 1

        # Convert the list to a TensorFlow Variable
        indexes_nonzero = tf.Variable(indexes_list, dtype=tf.int64)

        num_values = len(indexes_list)  #This will dynamically match the length of indexes_list

        if self.ones:
            # Step 1: Generate uniform random values between -1 and 1
            random_values = self.tf_rng.uniform(shape=(num_values,), minval=-1, maxval=1)

            # Step 2: Apply sign function to convert values to either -1.0 or 1.0
            signed_values = tf.sign(random_values)

            # Step 3: Scale the signed values by sigma
            values = signed_values * self.sigma


        else:
            values = self.tf_rng.uniform(
                (num_values,),
                minval=-self.sigma,
                maxval=self.sigma,
                dtype=dtype,
            )

        w_in = tf.SparseTensor(
            indices=indexes_nonzero, values=values, dense_shape=dense_shape
        )
        w_in = tf.sparse.reorder(w_in)
        w_in = tf.sparse.to_dense(w_in)
        return w_in

    def get_config(self) -> Dict:
        """Get the config dictionary of the initializer for serialization."""
        base_config = super().get_config()
        config = {"sigma": self.sigma, "ones": self.ones, "seed": self.seed}

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(package="MyInitializers", name="RegularNX")
class RegularNX(Initializer):
    """Regular graph adjacency matrix initializer.

    Generates a regular undirected graph's adjacency matrix with a given degree and spectral radius.
    The resulting matrix is symmetric and has a given spectral radius.

    Args:
        degree (int): Number of connections per node.
        spectral_radius (float): Spectral radius of the matrix.
        sigma (float): Standard deviation of the uniform distribution.
        ones (bool): If True, the matrix will be filled with ones.

    Returns:
        keras.initializers.Initializer: The initializer.

    Usage example.
    --------------

    >>> w_init = RegularNX(degree=3, spectral_radius=0.99, sigma=0.5, ones=False)
    >>> w = tf.Variable(w_init((10,10)))

    Usage example in a layer:
    -------------------------

    >>> w_init = RegularNX(degree=3, spectral_radius=0.45, sigma=0.5, ones=True)
    >>> layer = keras.layers.Dense(10, kernel_initializer=w_init)
    """

    def __init__(
        self,
        degree: int = 3,
        spectral_radius: float = 0.99,
        sigma: float = 0.5,
        ones: bool = False,
        seed: Union[int, None] = None,
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.ones = ones
        self.seed = seed

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def __call__(
        self,
        shape: Union[int, Tuple[int, int]],
        dtype: tf.dtypes.DType = tf.float32,
    ) -> tf.Tensor:
        """Generate the matrix.

        Args:
            shape (tuple): Shape of the matrix.

        Returns:
            tf.Tensor: The matrix.
        """
        if isinstance(shape, int):
            nodes = (shape, shape)
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and shape[0] == shape[1]
        ):
            nodes = tuple(shape)
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 equal integers"
            )

        # Make the amount of nodes at least the same size as the degree
        if nodes < self.degree:
            print(
                f"The number of nodes {nodes} is less than the degree {self.degree}, making the number of nodes equal to the degree."
            )
            nodes = self.degree

        # Check if n*d is even, else add 1 to n
        if nodes * self.degree % 2 != 0:
            print("nodes*degree is not even, adding 1 to nodes")
            nodes += 1

        degree = max(1, self.degree)
        graph = nx.random_regular_graph(degree, nodes, seed=self.rng)

        # Making non zero elements random uniform between -sigma and sigma
        for u, v in graph.edges():
            if not self.ones:  # Make this more efficient
                weight = self.rng.uniform(-self.sigma, self.sigma)
            else:
                weight = self.rng.choice([-1, 1])
            graph[u][v]["weight"] = weight

        # Convert to dense matrix to later on transform it to sparse matrix
        graph_matrix = nx.to_numpy_array(graph).astype(np.float32)

        # Go to sparse matrix to calculate the spectral radius efficiently
        graph_matrix = sparse.coo_matrix(graph_matrix)

        # Correcting the spectral radius
        print(f"Correcting spectral radius to {self.spectral_radius}")
        rho = abs(linalg.eigs(graph_matrix, k=1, which="LM")[0])[0]

        if rho == 0:
            print("The matrix is singular, re-initializing")
            return self(shape, dtype)

        kernel = graph_matrix * self.spectral_radius / rho

        print(f"Spectral radius was previously {rho}")

        # Converting to dense matrix
        kernel = kernel.toarray()

        # Casting to tensorflow
        kernel = tf.cast(kernel, dtype)

        return kernel

    def get_config(self) -> Dict:
        """Get the config dictionary of the initializer for serialization."""
        base_config = super().get_config()
        config = {
            "degree": self.degree,
            "spectral_radius": self.spectral_radius,
            "sigma": self.sigma,
            "ones": self.ones,
            "seed": self.seed,
        }

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(package="MyInitializers", name="ErdosRenyi")
class ErdosRenyi(Initializer):
    """Erdos Renyi adjacency matrix initializer.

    Uses networkx to generate a random graph and returns the adjacency matrix.

    Args:
        degree (int): Number of connections per node.
            Probability of connection is calculated as p = degree / (n-1).

        spectral_radius (float): Spectral radius of the matrix.

        sigma (float): Standard deviation of the uniform distribution.

        ones (bool): If True, the matrix will be filled with ones.

    Returns:
        keras.initializers.Initializer: The initializer.
    """

    def __init__(
        self,
        degree: int = 3,
        spectral_radius: float = 0.99,
        sigma: float = 0.5,
        ones: bool = False,
        seed: Union[int, None] = None,
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.ones = ones
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

    def __call__(
        self,
        shape: Union[int, Tuple[int, int]],
        dtype: tf.dtypes.DType = tf.float32,
    ) -> tf.Tensor:
        """Generate the matrix.

        Args:
            shape (int|tuple|list): Shape of the matrix.

        Returns:
            tf.Tensor: The matrix.

        """
        if isinstance(shape, int):
            nodes = shape
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and shape[0] == shape[1]
        ):
            nodes = shape[0]
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 equal integers"
            )

        # Make the amount of nodes at least the degree
        if nodes < self.degree:
            print(
                f"The number of nodes {nodes} is less than the degree {self.degree}, making the number of nodes equal to the degree."
            )
            nodes = self.degree

        # The probability of an Erdos-Renyi edge with degree <d> is p = d/(n-1)
        probab = self.degree / (nodes - 1)

        if probab < np.log(nodes) / nodes:
            print(
                f"The probability of connection is too low, p={probab}, should be greater than log(n)/n={np.log(nodes) / nodes}"
                f"it is probable the graph is disconnected. Increase the degree."
            )
            print("You ponder over life and the universe, and then continue...")

        graph = nx.erdos_renyi_graph(nodes, probab, directed=True, seed=self.rng)

        for u, v in graph.edges():
            if not self.ones:
                weight = self.rng.uniform(-self.sigma, self.sigma)
            else:
                weight = self.rng.choice([-1, 1])
            graph[u][v]["weight"] = weight

        # Convert to dense matrix to later on transform it to sparse matrix
        graph_matrix = nx.to_numpy_array(graph).astype(np.float32)

        # Convert to sparse matrix to calculate the spectral radius efficiently
        graph_matrix = sparse.coo_matrix(graph_matrix)

        print(f"Correcting spectral radius to {self.spectral_radius}")

        rho = abs(linalg.eigs(graph_matrix, k=1, which="LM")[0])[0]

        if rho == 0:
            print("The matrix is singular, re-initializing")
            return self(shape, dtype=dtype)

        print(f"Spectral radius was previously {rho}")

        kernel = graph_matrix * self.spectral_radius / rho

        # Convert back to dense matrix
        kernel = kernel.toarray()

        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self) -> Dict:
        """Get the config dictionary of the initializer for serialization."""
        base_config = super().get_config()
        config = {
            "degree": self.degree,
            "spectral_radius": self.spectral_radius,
            "sigma": self.sigma,
            "ones": self.ones,
            "seed": self.seed,
        }

        config.update(base_config)
        return config


@keras.saving.register_keras_serializable(package="MyInitializers", name="WattsStrogatzNX")
class WattsStrogatzNX(Initializer):
    """Watts Strogatz graph initializer.

    Uses networkx to generate the graph and extract the adjacency matrix of a
    Watts Strogatz graph. The generated graph is connected and undirected.

    Args:
        degree (int): The degree of the regular graph.
        spectral_radius (float): The spectral radius of the adjacency matrix.
        rewiring_p (float): The probability of rewiring each edge.
        sigma (float): The standard dviation of the weights.
        ones (bool): If True, the weights will be initialized to 1.

    Returns:
        keras.initializers.Initializer: The initializer.
    """

    def __init__(
        self,
        degree: int = 4,
        spectral_radius: float = 0.99,
        rewiring_p: float = 0.5,
        sigma: float = 0.5,
        ones: bool = False,
        seed: Union[int, None] = None,
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.rewiring_p = rewiring_p
        self.sigma = sigma
        self.ones = ones
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

    def __call__(
        self,
        shape: Union[int, Tuple[int, int]],
        dtype: tf.dtypes.DType = tf.float32,
    ) -> tf.Tensor:
        """Generate a Watts Strogatz graph adjacency matrix.

        Uses networkx to generate the graph and extract the adjacency matrix.

        Args:
            shape (tuple): The shape of the adjacency matrix.

        Returns:
            (np.array): The adjacency matrix.
        """
        if isinstance(shape, int):
            nodes = shape
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and shape[0] == shape[1]
        ):
            nodes = shape[0]
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 equal integers"
            )

        # Make the nodes at least the degree
        if nodes < self.degree:
            print(
                f"Number of nodes {nodes} is less than the degree {self.degree},"
                " making the number of nodes equal to the degree"
            )
            nodes = self.degree

        graph = nx.connected_watts_strogatz_graph(
            nodes, self.degree, self.rewiring_p, seed=self.rng
        )

        # Change the non zero values to random uniform in [-sigma, sigma]
        for u, v in graph.edges():
            if not self.ones:
                weight = self.rng.uniform(-self.sigma, self.sigma)
            else:
                weight = self.rng.choice([-1, 1])
            graph[u][v]["weight"] = weight

        # Convert to dense matrix to later on transform it to sparse matrix
        graph_matrix = nx.to_numpy_array(graph).astype(np.float32)

        # Going back to a sparse matrix to calculate the spectral radius efficiently
        graph_matrix = sparse.coo_matrix(graph_matrix)

        print(f"Correcting spectral radius to {self.spectral_radius}")

        rho = abs(
            linalg.eigs(
                graph_matrix,
                k=1,
                which="LM",
                return_eigenvectors=False,
                v0=np.ones(graph_matrix.shape[0]),
            )[0]
        )

        if rho == 0:
            print("The matrix is singular, re-initializing")
            return self(shape)

        print(f"Spectral radius was previously {rho}")

        kernel = graph_matrix * self.spectral_radius / rho

        # Convert back to dense matrix
        kernel = kernel.toarray()

        return tf.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self) -> Dict:
        """Get the config dictionary of the initializer for serialization."""
        base_config = super().get_config()
        config = {
            "degree": self.degree,
            "spectral_radius": self.spectral_radius,
            "rewiring_p": self.rewiring_p,
            "sigma": self.sigma,
            "ones": self.ones,
            "seed": self.seed,
        }

        config.update(base_config)
        return config

