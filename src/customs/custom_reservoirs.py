import os
# To eliminate tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict, Any

from keras import ops


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
    
    rule = np.flip(rule, axis=[0])
    
    num_neighbors = int((np.log2(len(rule)) - 1) / 2)
    
    assert num_neighbors == ((np.log2(len(rule)) - 1) / 2), "Rule length must be 2^n"


    powers = 2 ** np.arange(0, 2 * num_neighbors + 1, dtype="float32")
    powers = np.flip(powers)

    # @tf.function
    def apply_rule(triad: tf.Tensor) -> tf.Tensor:
        """Apply the rule based on the current state."""

        decimal_triad = ops.tensordot(triad, powers, axes=1)
        int_triad = ops.cast(decimal_triad, dtype="int32")
        
        if ops.equal(decimal_triad, ops.cast(int_triad, dtype="float32")):
            return ops.get_item(rule, int_triad)
        
        elif decimal_triad < ops.cast(int_triad, dtype="float32") + 0.5:
            return 2 * (ops.get_item(rule, int_triad + 1) - ops.get_item(rule, int_triad)) * ops.power((decimal_triad - ops.cast(int_triad, dtype="float32")), 2) + ops.get_item(rule, int_triad)
        
        else:
            return 2 * (ops.get_item(rule, int_triad) - ops.get_item(rule, int_triad + 1)) * ops.power((decimal_triad - ops.cast(int_triad, dtype="float32") - 1), 2) + ops.get_item(rule, int_triad + 1)

    # @tf.function
    def automaton(state_vector: tf.Tensor) -> tf.Tensor:
        """Run the automaton for the given number of steps."""
        assert steps > 0, "Number of steps must be positive"
    
        state_vector = ops.convert_to_tensor(state_vector, dtype="float32")
        state_vector = ops.reshape(state_vector, [-1])
        n = ops.size(state_vector)
        
        for _ in range(steps):
            new_state = tf.TensorArray(dtype=state_vector.dtype, size=n)
            for i in range(n):
                triad = ops.stack([state_vector[(i + j - num_neighbors) % n] for j in range(2 * num_neighbors + 1)])
                new_state = new_state.write(i, apply_rule(triad))
            state_vector = new_state.stack()
        return ops.reshape(state_vector, [1, -1])

    return automaton


def main():
    rule = np.array([i for i in "{0:08b}".format(110)], dtype=float)
    
    print("rule: ", rule)
    
    initial_state = np.random.randint(0, 2, (1,10)).astype(np.float32)
    
    print("initial state: ", initial_state)

    automaton = create_automaton_tf(rule, steps=1)
    
    print("final state: ", automaton(initial_state))


if __name__ == "__main__":
    main()
