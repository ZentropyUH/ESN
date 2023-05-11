import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.customs.custom_initializers import *


class Reservoir:
    def __init__(self, units):
        self.units = units

    # Define the structure of the reservoir here

    @tf.function
    def call(self, tensor):
        pass

    def __call__(self, tensor):
        return self.call(tensor)


class Dummy(Reservoir):
    def __init__(self, units, **kwargs):
        self.units = units

        self.w = RegularNX()((units, units))
        print(self.w)

        super().__init__(units, **kwargs)

    def call(self, inputs):
        return self.w * nputs
