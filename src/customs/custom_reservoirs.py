import tensorflow as tf

from src.customs.custom_initializers import *


#### Reservoir functions ####

def geca_reservoir(rule, steps=1):
    """Create an automaton from a rule.

    Args:
        rule (np.array): The rule to use. Elements should be float in [0,1].

        steps (int): The number of steps to run the automaton.

    Returns:
        function: The automaton function.
    """
    neighbors = (np.log2(rule.shape[0]) - 1) / 2

    assert neighbors == int(neighbors), "Rule must have length 2^n"

    neighbors = int(neighbors)
    print("neighbors: ", neighbors)

    def f(n):
        print(n)
        return rule[n]

    def automaton(state_vector):
        """Run the automaton for the given number of steps.

        Args:
            state_vector (np.array): The initial state of the automaton.

        Returns:
            np.array: The final state of the automaton.
        """
        # state_vector = state_vector[0]
        n = len(state_vector)
        next_state = np.zeros(n)

        print(len(state_vector) == len(next_state))

        for j in range(steps):
            for i in range(n):
                powers = np.flip(2 ** np.arange(2 * neighbors + 1))

                triad = [
                    state_vector[(i - neighbors + j) % n]
                    for j in range(2 * neighbors + 1)
                ]

                print("powers: ", powers)

                print("triad: ", triad)

                decimal_triad = np.dot(triad, powers)

                int_triad = int(decimal_triad)

                print("decimal_triad: ", decimal_triad)
                print("int_triad: ", int_triad)

                if decimal_triad == int_triad:
                    print("zero")
                    next_state[i] = f(int_triad)

                elif decimal_triad < int_triad + 0.5:
                    print("first")
                    next_state[i] = 2 * (f(int_triad + 1) - f(int_triad)) * (
                        decimal_triad - int_triad
                    ) ** 2 + f((int_triad))
                else:
                    print("second")
                    next_state[i] = 2 * (f(int_triad) - f(int_triad + 1)) * (
                        decimal_triad - int_triad - 1
                    ) ** 2 + f((int_triad + 1))
        return next_state

    return automaton


def main():
    # automaton = eca_generator("{0:08b}".format(8))

    rule = np.array([i for i in "{0:08b}".format(90)], dtype=float)
    print("rule: ", rule)

    # rule = np.random.choice((0, 1), 8)

    rule[0] = 0.3

    rule[1] = 0.5

    automaton = geca_reservoir(rule)
    print("second: ", automaton(np.array([1, 1, 1, 0, 0, 0, 1, 0, 1])))


if __name__ == "__main__":
    main()
