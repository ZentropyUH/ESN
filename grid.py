from src.grid.grid_script import *
import src.grid.grid_script
from src.grid.grid_one import *


# # The hyperparameters will be of the form: name: (initial_value, number_of_values, increment, function_of_increment)
# # The parameters of the increment function are: initial_value, increment, current_value_of_the_iteration
# hyperparameters_to_adjust = {
#     "sigma": (0.2, 5, 0.2, lambda x, y, i: round(x + y * i, 2)),
#     "degree": (2, 4, 2, lambda x, y, i: round(x + y * i, 2)),
#     "ritch_regularization": (10e-5, 5, 0.1, lambda x, y, i: round(x * y**i, 8)),
#     "spectral_radio": (0.9, 16 , 0.01, lambda x, y, i: round(x + y * i, 2)),
#     "reconection_prob": (0, 6, 0.2, lambda x, y, i: round(x + y*i, 2))
# }

# save_combinations(hyperparameters_to_adjust)
