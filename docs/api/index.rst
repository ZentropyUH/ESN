KRC API Reference
=================

.. toctree::
   :maxdepth: 2
   :hidden:


Core packages
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   keras_reservoir_computing.models.architectures
   keras_reservoir_computing.training.training

Layers
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   keras_reservoir_computing.layers
   keras_reservoir_computing.layers.reservoirs.layers.esn
   keras_reservoir_computing.layers.reservoirs.cells.base
   keras_reservoir_computing.layers.reservoirs.cells.esn_cell
   keras_reservoir_computing.layers.readouts.base
   keras_reservoir_computing.layers.readouts.ridge
   keras_reservoir_computing.layers.readouts.moorepenrose

Initializers
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   keras_reservoir_computing.initializers
   keras_reservoir_computing.initializers.helpers
   keras_reservoir_computing.initializers.input_initializers
   keras_reservoir_computing.initializers.recurrent_initializers
   keras_reservoir_computing.initializers.recurrent_initializers.graph_initializers

Utilities & IO
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   keras_reservoir_computing.utils
   keras_reservoir_computing.utils.data.analytics
   keras_reservoir_computing.utils.data.io
   keras_reservoir_computing.utils.general
   keras_reservoir_computing.utils.tensorflow
   keras_reservoir_computing.utils.visualization.graphics
   keras_reservoir_computing.utils.visualization.animations
   keras_reservoir_computing.io.loaders

HPO
---

.. autosummary::
   :toctree: generated
   :nosignatures:

   keras_reservoir_computing.hpo
   keras_reservoir_computing.hpo.main
   keras_reservoir_computing.hpo._objective
   keras_reservoir_computing.hpo._losses