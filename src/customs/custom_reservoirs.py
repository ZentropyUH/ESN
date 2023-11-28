import tensorflow as tf
import numpy as np
from typing import Callable, List, Tuple, Union, Optional, Dict, Any

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
        state_vector = tf.identity(state_vector)  # Ensure TensorFlow manages the state_vector tensor
        for _ in tf.range(steps):
            new_state = tf.TensorArray(dtype=state_vector.dtype, size=n)
            for i in tf.range(n):
                triad = tf.stack([state_vector[(i + j - num_neighbors) % n] for j in range(2 * num_neighbors + 1)])
                new_state = new_state.write(i, apply_rule(triad))
            state_vector = new_state.stack()
        return tf.reshape(state_vector, [1, -1])

    return automaton


def main():
    # automaton = eca_generator("{0:08b}".format(8))

    rule = np.array([i for i in "{0:08b}".format(110)], dtype=float)
    # print("rule: ", rule)

    # rule = np.random.choice((0, 1), 8)

    # rule[0] = 0.3

    rule[1] = 0.5

    # rule = "01101110"

    automaton = create_automaton_tf(rule)
    
    initial_state = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    
    print("initial: ", initial_state)
    
    print("second: ", automaton(initial_state))
    
    print("third: ", automaton(automaton(initial_state)))
    
    print("fourth: ", automaton(automaton(automaton(initial_state))))
    
    print("tst: ", automaton([0,0,1]) )


if __name__ == "__main__":
    main()
