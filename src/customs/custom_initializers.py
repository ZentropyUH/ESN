"""Custom Initializers."""
from typing import Dict, Union, List, Tuple

import networkx as nx
import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.sparse import linalg

import keras
import keras.utils
import keras.initializers
from keras.initializers import Initializer

###############################################
################## Initializers ###############
###############################################


@tf.keras.utils.register_keras_serializable(package="custom")
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

    def __init__(self, sigma: float = 0.5, **kwargs) -> None:
        """Initialize the initializer."""
        assert sigma > 0, "sigma must be positive"

        self.sigma = sigma
        super().__init__(**kwargs)

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
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(isinstance(d, int) for d in shape)
        ):
            rows, cols = tuple(shape)
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 integers"
            )

        dense_shape = (rows, cols)

        assert (
            rows <= cols
        ), "Reservoir nodes must be greater than or equal to number of features of the input."

        inputs_per_node = int(cols / rows)

        # There's a tweak for when cols/rows is not an integer,
        # hence it leaves the last column empty and that node is
        # not connected to the input

        # Correction to ensure at leat one connection per node
        q_flag = int(inputs_per_node < cols / rows)

        indexes_nonzero = tf.zeros(
            (rows * inputs_per_node + q_flag, 2), dtype=tf.int64
        )
        indexes_nonzero = tf.Variable(indexes_nonzero, dtype=tf.int64)

        for i in range(0, rows):
            for j in range(0, inputs_per_node + q_flag):
                indexes_nonzero[i * inputs_per_node + j, :].assign(
                    [
                        i,
                        i * inputs_per_node + j,
                    ]
                )

        values = tf.random.uniform(
            (rows * inputs_per_node + q_flag,),
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
        config = {"sigma": self.sigma}
        return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package="custom")
# class RandomUniform(Initializer):
#     """Random uniform matrix Initializer

#     Args:
#         sigma (float): Standard deviation of the uniform distribution.

#     Returns:
#         keras.initializers.Initializer: The initializer.
#     """

#     def __init__(self, sigma=0.5, **kwargs) -> None:
#         """Initialize the initializer."""
#         assert sigma > 0, "sigma must be positive"

#         self.sigma = sigma
#         super().__init__()

#     def __call__(self, shape, dtype=tf.float64, **kwargs) -> tf.Tensor:
#         """Generate the matrix.

#         Args:
#             shape (tuple): Shape of the matrix.
#             dtype (tf.dtype): Data type of the matrix.

#         Returns:
#             tf.Tensor: The matrix.
#         """
#         print()
#         print("shape: ", shape)
#         print("Shape type: ", type(shape))

#         if isinstance(shape, int):
#             rows, cols = shape, shape

#         elif isinstance(shape, tuple) or isinstance(shape, list):
#             if len(shape) == 1:
#                 rows, cols = shape[0], shape[0]
#             elif len(shape) == 2:
#                 rows, cols = shape
#         else:
#             raise ValueError("Shape must be int or tuple")

#         w = tf.random.uniform(
#             (rows, cols), minval=-self.sigma, maxval=self.sigma, dtype=dtype
#         )
#         return w

#     def get_config(self) -> Dict:
#         """Get the config dictionary of the initializer for serialization."""
#         base_config = super().get_config()
#         config = {"sigma": self.sigma}
#         return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package="custom")
# class RegularOwn(Initializer):
#     """Regular graph adjacency matrix initializer.

#     Generates a regular graph adjacency matrix with a given degree.
#     The generated graph is an undirected graph.

#     Args:
#         degree (int): Number of connections per node.

#         spectral_radius (float): Spectral radius of the matrix.

#         sigma (float): Standard deviation of the uniform distribution.

#         ones (bool): If True, the matrix will be filled with ones.

#     Returns:
#         keras.initializers.Initializer: The initializer.

#     Usage example.
#     -----------------

#     >>> w_init = RegularOwn(degree=3, spectral_radius=0.99, sigma=0.5, ones=False)
#     >>> w = tf.Variable(w_init((10,10)))
#     """

#     def __init__(
#         self,
#         degree: int = 3,
#         spectral_radius: float = 0.99,
#         sigma: float = 0.5,
#         ones: bool = False,
#         **kwargs,
#     ) -> None:
#         """Initialize the initializer."""
#         self.degree = degree
#         self.spectral_radius = spectral_radius
#         self.sigma = sigma
#         self.ones = ones
#         super().__init__(**kwargs)

#     def __call__(
#         self,
#         shape: Union[int, Tuple[int, int], List[int]],
#         dtype=tf.float32,
#     ) -> tf.Tensor:
#         """Generate the matrix.

#         Args:
#             shape (tuple): Shape of the matrix.

#         Returns:
#             tf.Tensor: The matrix.
#         """
#         # number of non-zero elements in the matrix is M*N*p
#         # number of non-zero elements per row is N*p
#         if isinstance(shape, int):
#             rows, cols = (shape, shape)
#         elif (
#             isinstance(shape, (list, tuple))
#             and len(shape) == 2
#             and all(isinstance(d, int) for d in shape)
#         ):
#             rows, cols = tuple(shape)
#         else:
#             raise ValueError(
#                 "Shape must be an integer or a tuple/list of 2 integers"
#             )

#         dense_shape = (rows, cols)

#         assert rows == cols, "Must be a square matrix."
#         assert (
#             self.degree <= cols
#         ), f"The nuber of connections: {self.degree} must be less or equal to the number of nodes {cols}."

#         degree = max(1, self.degree)

#         indices = np.zeros((rows * degree, 2), dtype=int)
#         for i in range(rows):
#             # Choose the columns of the non-zero elements in this row
#             # We are guaranteeing that the matrix is regular in this step
#             row_cols = np.random.choice(cols, size=degree, replace=False)
#             # Assign the indices of the non-zero elements
#             for j in range(degree):
#                 indices[i * degree + j] = [i, row_cols[j]]

#         if self.ones:
#             values = tf.ones((rows * degree,), dtype=dtype)
#         else:
#             values = tf.random.uniform((rows * degree,), minval=-self.sigma, maxval=self.sigma, dtype=dtype)

#         # Converting to scipy sparse to be able to calculate the spectral radius efficiently
#         print("Converting to scipy sparse matrix")
#         w_coo = sparse.coo_matrix((values, indices.T), shape=dense_shape)

#         print(f"Correcting spectral radius to {self.spectral_radius}")
#         rho = abs(linalg.eigs(w_coo, k=1, which="LM")[0])[0]

#         # Improve this, but it will usually happen only when the matrix is small
#         if rho == 0:
#             print("The matrix is singular, re-initializing")
#             return self(shape, dtype)

#         print(f"Spectral radius was previously {rho}")
#         print()

#         w_coo = w_coo * self.spectral_radius / rho

#         # Converting the matrix to tensorflow sparse matrix
#         indices = np.stack((w_coo.row, w_coo.col), axis=1)
#         sparse_w = tf.SparseTensor(indices, w_coo.data, dense_shape)
#         sparse_w = tf.sparse.reorder(sparse_w)

#         dense_w = tf.sparse.to_dense(sparse_w)

#         # Casting to dtype
#         dense_w = tf.cast(dense_w, dtype)

#         return dense_w

#     def get_config(self) -> Dict:
#         """Get the config dictionary of the initializer for serialization."""
#         base_config = super().get_config()
#         config = {
#             "degree": self.degree,
#             "spectral_radius": self.spectral_radius,
#             "sigma": self.sigma,
#             "ones": self.ones,
#         }
#         return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="custom")
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
        self, degree=3, spectral_radius=0.99, sigma=0.5, ones=False, **kwargs
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.ones = ones
        super().__init__(**kwargs)

    def __call__(self, shape, dtype=tf.float32) -> tf.Tensor:
        """Generate the matrix.

        Args:
            shape (tuple): Shape of the matrix.

        Returns:
            tf.Tensor: The matrix.
        """
        if isinstance(shape, int):
            rows, _cols = (shape, shape)
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(isinstance(d, int) for d in shape)
        ):
            rows, _cols = tuple(shape)
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 integers"
            )

        assert rows == _cols, "Must be a square matrix."

        # Make the amount of nodes at least the same size as the degree
        if rows < self.degree:
            print(
                f"The number of nodes {rows} is less than the degree {self.degree}, making the number of nodes equal to the degree."
            )
            rows = self.degree
            _cols = self.degree

        # Check if n*d is even, else add 1 to n
        if rows * self.degree % 2 != 0:
            print("nodes*degree is not even, adding 1 to nodes")
            rows += 1
            _cols += 1

        degree = max(1, self.degree)
        graph = nx.random_regular_graph(degree, rows)

        # Converting to numpy array to be able to change the values
        graph_matrix = nx.to_numpy_array(graph).astype(np.float32)

        # Making non zero elements random uniform between -sigma and sigma
        if not self.ones:  # Make this more efficient
            graph_matrix[graph_matrix != 0] = np.random.uniform(
                -self.sigma, self.sigma, graph_matrix[graph_matrix != 0].shape
            )

        # Go to sparse matrix to calculate the spectral radius efficiently
        graph_matrix = sparse.coo_matrix(graph_matrix)

        # Correcting the spectral radius
        print(f"Correcting spectral radius to {self.spectral_radius}")
        if self.ones:
            # If matrix is binary the spectral radius is the degree of the graph.
            rho = degree
            kernel = graph_matrix * self.spectral_radius / rho
        else:
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
        }
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="custom")
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
        self, degree=3, spectral_radius=0.99, sigma=0.5, ones=False, **kwargs
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.ones = ones
        super().__init__(**kwargs)

    def __call__(self, shape, dtype=tf.float32) -> tf.Tensor:
        """Generate the matrix.

        Args:
            shape (tuple|int): Shape of the matrix.

        Returns:
            tf.Tensor: The matrix.

        """
        if isinstance(shape, int):
            rows, cols = (shape, shape)
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(isinstance(d, int) for d in shape)
        ):
            rows, cols = tuple(shape)
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 integers"
            )

        assert rows == cols, "Matrix must be square"

        # Make the amount of nodes at least the degree
        if rows < self.degree:
            print(
                f"The number of nodes {rows} is less than the degree {self.degree}, making the number of nodes equal to the degree."
            )
            rows = self.degree
            cols = self.degree

        # The average degree in this model is n * p, where p is the probability of connection
        # and n is the number of nodes.
        probab = self.degree / rows

        if probab < np.log(rows) / rows:
            print(
                "The probability of connection is too low, "
                "it is probable the graph is disconnected. Increase the degree."
            )
            print(
                "You ponder over life and the universe, and then continue..."
            )

        graph = nx.erdos_renyi_graph(rows, probab, directed=True)

        if not self.ones:
            for u, v in graph.edges():
                weight = np.random.uniform(-self.sigma, self.sigma)
                graph[u][v]["weight"] = weight

        # Convert to dense matrix to make non zero values random uniform
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
        }
        return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package="custom")
# class WattsStrogatzOwn(Initializer):
#     """Generate a Watts Strogatz graph adjacency matrix.

#     Makes a regular graph adjacency matrix and then randomly rewire each edge
#     with probability rewiring_p.
#     The degree of each node is K, where K is even, otherwise it is rounded
#     up to the nearest even number.
#     When the probability of rewiring is 0, the graph is regular.
#     On the other hand, when the probability of rewiring is 1, the graph is closer
#     to an Erdos-Renyi graph.

#     Args:
#         degree (int): The degree of each node.

#         spectral_radius (float): Spectral radius of the matrix.

#         rewiring_p (float): The probability of rewiring each edge.

#         sigma (float): Standard deviation of the uniform distribution.

#         ones (bool): If True, the matrix will be filled with ones.

#     Returns:
#         keras.initializers.Initializer: The initializer.

#     Usage example:
#     --------------

#     >>> w_init = WattsStrogatzOwn(4, 0.5)
#     >>> w = tf.Variable(w_init(shape=(3, 3)))
#     """

#     @staticmethod
#     def regular_graph(nodes, degree, sigma=0.5) -> np.ndarray:
#         """
#         Generate a regular graph adjacency matrix.

#         Each node is connected to the `degree' nearest neighbors.

#         Args:
#             nodes (int): The number of nodes.

#             degree (int): The degree of each node.

#         Returns:
#             (np.array): The adjacency matrix.
#         """
#         # if degree > nodes - 1 set degree to nodes - 1
#         degree = nodes if degree > nodes - 1 else degree

#         graph = np.zeros((nodes, nodes), dtype=float)

#         for i in range(nodes):
#             for j in range(degree // 2):
#                 graph[i, (i + j + 1) % nodes] = np.random.uniform(
#                     -sigma, sigma
#                 )
#                 graph[i, (i - j - 1) % nodes] = np.random.uniform(
#                     -sigma, sigma
#                 )

#         return graph

#     @staticmethod
#     def watts_strogatz(graph, rewiring_p, sigma=0.5) -> nx.Graph:
#         """
#         Generate a Watts Strogatz graph adjacency matrix from a regular graph adjacency matrix.

#         Args:
#             shape (tuple): The shape of the adjacency matrix.
#             rewiring_p (float): The probability of rewiring each edge.

#         Returns:
#             (np.array): The adjacency matrix.
#         """
#         nodes = graph.shape[0]

#         for node_1 in range(nodes):
#             for node_2 in range(node_1 + 1, nodes):
#                 if (
#                     graph[node_1, node_2] != 0
#                     and np.random.rand() < rewiring_p
#                 ):
#                     graph[node_1, node_2] = 0

#                     # Comment the line below if the graph is directed
#                     # graph[node_2, node_1] = 0
#                     while True:
#                         new_j = np.random.randint(nodes)
#                         if graph[node_1, new_j] == 0:
#                             graph[node_1, new_j] = np.random.uniform(
#                                 -sigma, sigma
#                             )

#                             # Comment the line below if the graph is directed
#                             # graph[new_j, node_1] = np.random.uniform(
#                             #     -sigma, sigma
#                             # )
#                             break
#             # Remove self loops
#             graph[node_1, node_1] = 0
#         return graph

#     def __init__(
#         self,
#         degree=4,
#         spectral_radius=0.99,
#         rewiring_p=0.5,
#         sigma=0.5,
#         ones=False,
#         **kwargs
#     ) -> None:
#         """Initialize the initializer."""
#         if degree % 2 != 0:
#             degree += 1
#             print(
#                 f"Degree is not even, rounding up to nearest even number. New degree: {degree}"
#             )

#         self.degree = degree
#         self.spectral_radius = spectral_radius
#         self.rewiring_p = rewiring_p
#         self.sigma = sigma
#         self.ones = ones
#         super().__init__(**kwargs)

#     def __call__(self, shape, dtype=tf.float32) -> tf.Tensor:
#         """Generate a Watts Strogatz graph adjacency matrix.

#         Args:
#             shape (tuple): The shape of the adjacency matrix.

#         Returns:
#             (np.array): The adjacency matrix.
#         """
#         if isinstance(shape, int):
#             rows, cols = (shape, shape)
#         elif (
#             isinstance(shape, (list, tuple))
#             and len(shape) == 2
#             and all(isinstance(d, int) for d in shape)
#         ):
#             rows, cols = tuple(shape)
#         else:
#             raise ValueError(
#                 "Shape must be an integer or a tuple/list of 2 integers"
#             )

#         assert rows == cols, "Matrix must be square"

#         # make nodes at least 3
#         rows = max(3, rows)
#         cols = max(3, cols)

#         graph = self.regular_graph(rows, self.degree, sigma=self.sigma)
#         ws_graph = self.watts_strogatz(
#             graph, self.rewiring_p, sigma=self.sigma
#         )

#         # Guarantee that the graph is connected
#         nx_graph = nx.from_numpy_array(ws_graph)
#         connected = nx.is_connected(nx_graph)

#         # Make this a parameter later
#         iterations = 0
#         while not connected:
#             ws_graph = self.watts_strogatz(graph, self.rewiring_p)

#             nx_graph = nx.from_numpy_array(ws_graph)
#             connected = nx.is_connected(nx_graph)

#             iterations += 1

#             if iterations > 100:
#                 raise StopIteration(
#                     "Could not generate a connected graph. "
#                     "Try increasing the number of nodes or decreasing the rewiring probability."
#                 )

#         # Correct the spectral radius

#         # convert to sparse matrix
#         print(f"Correcting spectral radius to {self.spectral_radius}")

#         # Make non-zero elements ones
#         if self.ones:
#             ws_graph[ws_graph != 0] = 1

#         # Convert to sparse matrix to calculate the spectral radius efficiently
#         ws_graph = sparse.csr_matrix(ws_graph)

#         rho = max(abs(linalg.eigs(ws_graph, k=1, which="LM")[0]))

#         ws_graph = ws_graph / rho * self.spectral_radius

#         print(f"Spectral radius was previously {rho}")

#         kernel = ws_graph.toarray()

#         return tf.convert_to_tensor(kernel, dtype=dtype)

#     def get_config(self) -> Dict:
#         """Get the configuration of the initializer."""
#         return {
#             "degree": self.degree,
#             "rewiring_p": self.rewiring_p,
#             "sigma": self.sigma,
#         }


@tf.keras.utils.register_keras_serializable(package="Custom")
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
        degree=4,
        spectral_radius=0.99,
        rewiring_p=0.5,
        sigma=0.5,
        ones=False,
        **kwargs,
    ) -> None:
        """Initialize the initializer."""
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.rewiring_p = rewiring_p
        self.sigma = sigma
        self.ones = ones
        super().__init__(**kwargs)

    def __call__(self, shape, dtype=tf.float32) -> tf.Tensor:
        """Generate a Watts Strogatz graph adjacency matrix.

        Uses networkx to generate the graph and extract the adjacency matrix.

        Args:
            shape (tuple): The shape of the adjacency matrix.

        Returns:
            (np.array): The adjacency matrix.
        """
        if isinstance(shape, int):
            rows, cols = (shape, shape)
        elif (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(isinstance(d, int) for d in shape)
        ):
            rows, cols = tuple(shape)
        else:
            raise ValueError(
                "Shape must be an integer or a tuple/list of 2 integers"
            )

        assert rows == cols, "Matrix must be square"

        # Make the nodes at least the degree
        if rows < self.degree:
            print(
                f"Number of nodes {rows} is less than the degree {self.degree},"
                " making the number of nodes equal to the degree"
            )
            rows = self.degree
            cols = self.degree

        graph = nx.connected_watts_strogatz_graph(
            rows, self.degree, self.rewiring_p
        )

        # Change the non zero values to random uniform in [-sigma, sigma]
        if not self.ones:
            # Convert to numpy array and change non zero values
            graph_matrix = nx.to_numpy_array(graph)
            # make non zero values randomly uniform in [-sigma, sigma]
            graph_matrix[graph_matrix != 0] = np.random.uniform(
                -self.sigma, self.sigma, graph_matrix[graph_matrix != 0].shape
            )
            # Going back to a sparse matrix to calculate the spectral radius efficiently
            graph_matrix = sparse.csr_matrix(graph_matrix)

        print(f"Correcting spectral radius to {self.spectral_radius}")

        rho = abs(linalg.eigs(graph_matrix, k=1, which="LM")[0])[0]

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
            "sigma": self.sigma,
        }
        return dict(list(base_config.items()) + list(config.items()))


custom_initializers = {
    "InputMatrix": InputMatrix,
    "RegularNX": RegularNX,
    "ErdosRenyi": ErdosRenyi,
    "WattsStrogatzNX": WattsStrogatzNX,
    "Zeros": keras.initializers.Zeros,
    "RandomUniform": keras.initializers.RandomUniform
}

keras.utils.get_custom_objects().update(custom_initializers)
