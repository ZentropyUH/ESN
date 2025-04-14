from typing import Optional

import numpy as np
import tensorflow as tf
from scipy.sparse.csgraph import connected_components

from keras_reservoir_computing.initializers.helpers import spectral_radius_hybrid


@tf.keras.utils.register_keras_serializable(
    package="krc", name="ConnectedRandomMatrixInitializer"
)
class ConnectedRandomMatrixInitializer(tf.keras.Initializer):
    """
    Initializer for creating random matrices with guaranteed connectivity.

    This initializer generates a random matrix with values between -a and a,
    with a specified density of non-zero values, ensuring the associated
    graph is connected.

    Parameters
    ----------
    max_value : float, default=1.0
        Maximum absolute value of matrix entries.
    density : float, default=0.3
        Float in [0,1] representing the target proportion of non-zero entries.
    directed : bool, default=True
        Whether the associated graph should be directed.
    spectral_radius : float, optional
        If provided, the matrix will be scaled to have this spectral radius.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tf.Tensor
        A random connected matrix with the specified properties.

    Notes
    -----
    - Connectivity ensures there exists a path between any pair of nodes in the graph.
    - Minimum density is automatically calculated based on graph size to ensure connectivity.
    - Self-loops are allowed in the generated matrices.

    Examples
    --------
    >>> from keras_reservoir_computing.initializers import ConnectedRandomMatrixInitializer
    >>> initializer = ConnectedRandomMatrixInitializer(max_value=10, density=0.3, spectral_radius=0.9)
    >>> W = initializer((100, 100))
    >>> print(f"Matrix shape: {W.shape}")
    >>> print(f"Density: {tf.math.count_nonzero(W).numpy() / (100*100):.4f}")
    """

    def __init__(
        self,
        max_value: float = 1.0,
        density: float = 0.3,
        directed: bool = True,
        spectral_radius: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the initializer with specified parameters."""
        self.max_value = max_value
        self.density = density
        self.directed = directed
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple, dtype=None) -> tf.Tensor:
        """
        Generate a random connected matrix with the specified shape.

        Parameters
        ----------
        shape : tuple
            Shape of the matrix to create. Must be (N, N) for a square matrix.
        dtype : tf.DType, optional
            Data type for the generated matrix.

        Returns
        -------
        tf.Tensor
            The initialized weight matrix of shape (N, N).

        Raises
        ------
        ValueError
            If shape is not 2D or not square, or if density is too low for connectivity.
        """
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Shape must be (N, N), got {shape}")

        n = shape[0]
        # Use float32 as the default dtype if not specified
        if dtype is None:
            dtype = tf.float32

        # Generate the matrix
        matrix = self._random_connected_matrix(
            n, self.max_value, self.density, self.directed
        )

        # Explicitly convert to float32 before tensor conversion
        matrix = matrix.astype(np.float32)
        matrix = tf.convert_to_tensor(matrix, dtype=dtype)

        if self.spectral_radius is not None:
            try:
                sr = spectral_radius_hybrid(matrix)
                if sr > 0:  # Avoid division by zero
                    matrix = matrix * (self.spectral_radius / sr)
                else:
                    tf.debugging.assert_greater(
                        sr, 0.0, 
                        message="Spectral radius calculation returned zero or negative value."
                    )
            except Exception as e:
                tf.print(f"Warning: Spectral radius calculation failed. Using matrix without scaling. Error: {e}")

        return matrix

    def _random_connected_matrix(self, n, a, density, directed):
        """
        Generate a random nxn matrix with values between -a and a,
        with density of non-zero values controlled by parameter in [0,1],
        ensuring the associated graph is connected.
        """
        if n <= 0 or a <= 0 or not 0 <= density <= 1:
            raise ValueError(
                "Invalid parameters: n and a must be positive, density must be in [0,1]"
            )

        # Handle n=1 case
        if n == 1:
            return np.array(
                [[self._random_nonzero_value(a) if self.rng.random() < density else 0]]
            )

        # Calculate minimum edges needed for connectivity
        min_edges = n if directed else (n - 1)

        # Calculate total possible entries (including diagonal)
        total_possible_edges = n * n if directed else n * (n + 1) // 2

        # Check if density is sufficient for connectivity
        min_density = min_edges / total_possible_edges
        if density < min_density:
            raise ValueError(
                f"Density must be at least {min_density:.6f} to ensure connectivity"
            )

        # Generate matrix and ensure connectivity
        matrix = self._create_random_sparse_matrix(n, a, density, directed)
        return self._ensure_connectivity(matrix, a, directed)

    def _create_random_sparse_matrix(self, n, a, density, directed):
        """Create a random sparse matrix with given density, including self-loops."""
        matrix = np.zeros((n, n))

        # Calculate total possible entries (including diagonal)
        total_possible_edges = n * n if directed else n * (n + 1) // 2

        # Calculate number of edges to include
        num_edges = int(density * total_possible_edges)

        # Generate all possible edge positions (including diagonal)
        if directed:
            possible_edges = [(i, j) for i in range(n) for j in range(n)]
        else:
            possible_edges = [(i, j) for i in range(n) for j in range(i, n)]

        # Randomly select edges and assign values
        for i, j in self.rng.choice(
            possible_edges, size=min(num_edges, len(possible_edges)), replace=False
        ):
            value = self._random_nonzero_value(a)
            matrix[i, j] = value
            if not directed and i != j:
                matrix[j, i] = value  # Symmetric for undirected

        return matrix

    def _ensure_connectivity(self, matrix, a, directed):
        """Ensure the matrix represents a connected graph while preserving density."""
        n = matrix.shape[0]
        n_components = connected_components(
            matrix, directed=directed, return_labels=False
        )

        if n_components == 1:
            return matrix  # Already connected

        # Get connected components and organize nodes by component
        _, labels = connected_components(matrix, directed=directed, return_labels=True)
        component_sets = [
            set(i for i in range(n) if labels[i] == comp)
            for comp in range(n_components)
        ]

        # Add edges to connect components
        edges_to_add = []
        for comp1 in range(n_components - 1):
            # Connect components with random nodes
            node1 = self.rng.choice(list(component_sets[comp1]))
            node2 = self.rng.choice(list(component_sets[comp1 + 1]))

            edges_to_add.append((node1, node2))
            if not directed:
                edges_to_add.append((node2, node1))

        # Add connecting edges
        for i, j in edges_to_add:
            matrix[i, j] = self._random_nonzero_value(a)

        # If needed, remove edges to maintain density
        edges_added = len(edges_to_add)
        if edges_added > 0:
            # Find non-zero entries and shuffle them
            nonzero_indices = list(zip(*np.nonzero(matrix)))
            self.rng.shuffle(nonzero_indices)

            # Remove edges that don't break connectivity
            for i, j in nonzero_indices:
                temp_matrix = matrix.copy()
                temp_matrix[i, j] = 0

                if (
                    connected_components(
                        temp_matrix, directed=directed, return_labels=False
                    )
                    == 1
                ):
                    matrix[i, j] = 0
                    edges_added -= 1
                    if edges_added <= 0:
                        break

        return matrix

    def _random_nonzero_value(self, a):
        """Generate a random non-zero value between -a and a."""
        return self.rng.uniform(-a, a) or 1e-10  # Fallback if we hit exactly zero

    def get_config(self) -> dict:
        """
        Get the config dictionary of the initializer for serialization.

        Returns
        -------
        dict
            The configuration dictionary.
        """
        config = {
            "max_value": self.max_value,
            "density": self.density,
            "directed": self.directed,
            "spectral_radius": self.spectral_radius,
            "seed": self.seed,
        }
        base_config = super().get_config()
        base_config.update(config)
        return config
