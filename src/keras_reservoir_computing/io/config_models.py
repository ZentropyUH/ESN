"""Pydantic models for configuration validation and management.

This module provides Pydantic models for validating and managing configurations
for reservoir computing models, including reservoir layers and readout layers.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class LayerConfig(BaseModel):
    """Base configuration model for Keras layers.

    This model provides a base structure for layer configurations that can be
    serialized to/from JSON/YAML and validated using Pydantic.

    Parameters
    ----------
    class_name : str
        The class name of the layer (e.g., "krc>ESNReservoir").
    config : Dict[str, Any]
        Configuration dictionary for the layer.

    Attributes
    ----------
    class_name : str
        The class name of the layer.
    config : Dict[str, Any]
        Configuration dictionary for the layer.
    """

    class_name: str = Field(
        ..., description="The class name of the layer (e.g., 'krc>ESNReservoir')."
    )
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration dictionary for the layer."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {"class_name": self.class_name, "config": self.config}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerConfig":
        """Create a LayerConfig from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing class_name and config.

        Returns
        -------
        LayerConfig
            LayerConfig instance.
        """
        return cls(**data)


class ReservoirConfig(LayerConfig):
    """Configuration model for reservoir layers.

    This model validates and manages configurations for reservoir layers,
    ensuring that required parameters are present and properly typed.

    Parameters
    ----------
    class_name : str, optional
        The class name of the reservoir layer. Default is "krc>ESNReservoir".
    config : Dict[str, Any], optional
        Configuration dictionary for the reservoir layer. Should contain
        parameters like units, feedback_dim, activation, initializers, etc.

    Attributes
    ----------
    class_name : str
        The class name of the reservoir layer.
    config : Dict[str, Any]
        Configuration dictionary for the reservoir layer.

    Notes
    -----
    The config dictionary should contain the following keys (depending on
    the reservoir type):
    - units: int - Number of units in the reservoir
    - feedback_dim: int - Dimensionality of the feedback input
    - input_dim: int - Dimensionality of the external input
    - leak_rate: float - Leaking rate of the reservoir neurons
    - activation: str or callable - Activation function
    - input_initializer: dict - Input weight initializer config
    - feedback_initializer: dict - Feedback weight initializer config
    - feedback_bias_initializer: dict - Feedback bias initializer config
    - kernel_initializer: dict - Recurrent weight initializer config
    - dtype: str - Data type for the layer
    """

    class_name: str = Field(
        default="krc>ESNReservoir", description="The class name of the reservoir layer."
    )

    def update_config(self, **kwargs: Any) -> "ReservoirConfig":
        """Update configuration with new values.

        This method creates a new ReservoirConfig with updated values.
        It only adds keys that don't already exist in the config; it does
        not override existing config values.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs to add to the config dictionary if not present.

        Returns
        -------
        ReservoirConfig
            New ReservoirConfig instance with updated values.
        """
        new_config = self.config.copy()
        for key, value in kwargs.items():
            if key not in new_config:
                new_config[key] = value
        return ReservoirConfig(class_name=self.class_name, config=new_config)

    def override_config(self, **kwargs: Any) -> "ReservoirConfig":
        """Override configuration with new values.

        This method creates a new ReservoirConfig with overridden values.
        Unlike update_config, this will replace existing values.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs to override in the config dictionary.

        Returns
        -------
        ReservoirConfig
            New ReservoirConfig instance with overridden values.
        """
        new_config = self.config.copy()
        for key, value in kwargs.items():
            new_config[key] = value
        return ReservoirConfig(class_name=self.class_name, config=new_config)


class ReadoutConfig(LayerConfig):
    """Configuration model for readout layers.

    This model validates and manages configurations for readout layers,
    ensuring that required parameters are present and properly typed.

    Parameters
    ----------
    class_name : str, optional
        The class name of the readout layer. Default is "krc>RidgeReadout".
    config : Dict[str, Any], optional
        Configuration dictionary for the readout layer. Should contain
        parameters like units, alpha, max_iter, tol, etc.

    Attributes
    ----------
    class_name : str
        The class name of the readout layer.
    config : Dict[str, Any]
        Configuration dictionary for the readout layer.

    Notes
    -----
    The config dictionary should contain the following keys (depending on
    the readout type):
    - units: int - Number of output units
    - alpha: float - L2 regularization strength (for RidgeReadout)
    - max_iter: int - Maximum iterations for solver (for RidgeReadout)
    - tol: float - Tolerance for solver (for RidgeReadout)
    - trainable: bool - Whether the layer is trainable
    - dtype: str - Data type for the layer
    """

    class_name: str = Field(
        default="krc>RidgeReadout", description="The class name of the readout layer."
    )

    def update_config(self, **kwargs: Any) -> "ReadoutConfig":
        """Update configuration with new values.

        This method creates a new ReadoutConfig with updated values.
        It only adds keys that don't already exist in the config; it does
        not override existing config values.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs to add to the config dictionary if not present.

        Returns
        -------
        ReadoutConfig
            New ReadoutConfig instance with updated values.
        """
        new_config = self.config.copy()
        for key, value in kwargs.items():
            if key not in new_config:
                new_config[key] = value
        return ReadoutConfig(class_name=self.class_name, config=new_config)

    def override_config(self, **kwargs: Any) -> "ReadoutConfig":
        """Override configuration with new values.

        This method creates a new ReadoutConfig with overridden values.
        Unlike update_config, this will replace existing values.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs to override in the config dictionary.

        Returns
        -------
        ReadoutConfig
            New ReadoutConfig instance with overridden values.
        """
        new_config = self.config.copy()
        for key, value in kwargs.items():
            new_config[key] = value
        return ReadoutConfig(class_name=self.class_name, config=new_config)
