import os
# To eliminate tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict, Any

from time import time

# from src.customs.custom_initializers import *


#### Reservoir functions ####

def create_automaton_tf(rule: Union[int, str, np.ndarray, tf.Tensor], steps: int = 1) -> Callable:
    """Create a cellular automaton from a given rule using TensorFlow.

    Args:
        rule (Union[int, str, np.ndarray, tf.Tensor]): The rule to be used for the automaton.
        steps (int, optional): The number of steps to be used for the automaton. Defaults to 1.

    Returns:
        Callable: The automaton function.
    """
    
    if isinstance(rule, int):
        rule = np.array([i for i in "{0:08b}".format(rule)], dtype=float)
    
    if isinstance(rule, str):
        rule = np.array([i for i in rule], dtype=float)
    
    rule = tf.convert_to_tensor(rule, dtype=tf.float32)
    rule = tf.reverse(rule, axis=[0])
    
    num_neighbors = int((tf.math.log(tf.cast(tf.size(rule), tf.float32)) / tf.math.log(2.0) - 1) / 2)
    assert num_neighbors == ((tf.math.log(tf.cast(tf.size(rule), tf.float32)) / tf.math.log(2.0) - 1) / 2), "Rule length must be 2^n"
    
    powers = tf.math.pow(2, tf.range(0, 2 * num_neighbors + 1, dtype=tf.float32))
    powers = tf.reverse(powers, axis=[0])

    @tf.function
    def apply_rule(triad: tf.Tensor) -> tf.Tensor:
        """Apply the rule based on the current state."""
                
        decimal_triad = tf.tensordot(triad, powers, axes=1)
        int_triad = tf.cast(decimal_triad, tf.int32)
        
        if decimal_triad == tf.cast(int_triad, tf.float32):
            return tf.gather(rule, int_triad)
        
        elif decimal_triad < tf.cast(int_triad, tf.float32) + 0.5:
            return 2 * (tf.gather(rule, int_triad + 1) - tf.gather(rule, int_triad)) * tf.math.pow((decimal_triad - tf.cast(int_triad, tf.float32)), 2) + tf.gather(rule, int_triad)
        
        else:
            return 2 * (tf.gather(rule, int_triad) - tf.gather(rule, int_triad + 1)) * tf.math.pow((decimal_triad - tf.cast(int_triad, tf.float32) - 1), 2) + tf.gather(rule, int_triad + 1)

    @tf.function
    def automaton(state_vector: tf.Tensor) -> tf.Tensor:
        """Run the automaton for the given number of steps."""
        assert steps > 0, "Number of steps must be positive"
    
        state_vector = tf.convert_to_tensor(state_vector, dtype=tf.float32)

        state_vector = tf.reshape(state_vector, [-1])
        
        n = tf.size(state_vector)

        
        new_state_vector = tf.TensorArray(dtype=state_vector.dtype, size=n)
        for i in range(n):
            if state_vector[i] < 0:
                # print("Value less than 0: ", state_vector[i])
                new_state_vector = new_state_vector.write(i, 0)
            elif state_vector[i] > 1:
                # print("Value greater than 1: ", state_vector[i])
                new_state_vector = new_state_vector.write(i, 1)
            else:
                new_state_vector = new_state_vector.write(i, state_vector[i])
        state_vector = new_state_vector.stack()

        
        state_vector = tf.identity(state_vector)  # Ensure TensorFlow manages the state_vector tensor
        for _ in tf.range(steps):
            new_state = tf.TensorArray(dtype=state_vector.dtype, size=n)
            for i in tf.range(n):
                triad = tf.stack([state_vector[(i + j - num_neighbors) % n] for j in range(2 * num_neighbors + 1)])
                # init_time = time()
                new_state = new_state.write(i, apply_rule(triad))
                # print("rule: ", time()-init_time)
            state_vector = new_state.stack()
        # print("ECA: ", time()-init_time)
        return tf.reshape(state_vector, [1, -1])

    return automaton


def main():
    rule = np.array([i for i in "{0:08b}".format(110)], dtype=float)

    automaton = create_automaton_tf(rule, steps=1)

    initial_state = np.random.rand(1, 1000).astype(np.float32)

    # initial_state[0, 100] = 2
    a = automaton(initial_state)


if __name__ == "__main__":
    main()
