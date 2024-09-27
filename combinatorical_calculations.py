"""
MIT License
Copyright © 2024 David A. Reiss
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""


import numpy as np
from itertools import product
from collections.abc import Iterable


class CombinatoricalCalculations:
    def __init__(self, eps=0.):
        prob_step_up = 0.5 + eps
        prob_step_down = 1 - prob_step_up

    @staticmethod
    def generate_all_trajectories(T: int):
        def convert_to_number_array(tup: tuple[str, ...]):
            return np.array([+ 1 if char == "+" else - 1 for char in tup])

        trajectories_array_steps = np.zeros((2 ** T, T))
        trajectories_array_x = np.zeros((2 ** T, T))

        j = 0
        for trajectory in product("+-", repeat=T):
            trajectories_array_steps[j] = convert_to_number_array(trajectory)

            trajectories_array_x[j] = np.array([np.sum(trajectories_array_steps[j, :(t+1)])
                                                for t in range(T)])
            # converts information about steps # into x_t-coordinates

            j += 1

        return trajectories_array_steps, trajectories_array_x
        # 0. axis: all different trajectories
        # 1. axis: different steps at time t / x_t-values for respective trajectory


    @staticmethod
    def calculate_all_trajectory_probs(trajectories_array_x: np.ndarray, prob_step_up: float):
        T = np.shape(trajectories_array_x)[1]

        trajectories_array_no_up_steps = np.array([(t + trajectories_array_x[:, t]) / 2
                                                   for t in range(T)]).transpose()
        trajectories_array_no_down_steps = np.array([(t - trajectories_array_x[:, t]) / 2
                                                     for t in range(T)]).transpose()

        prob_step_down = 1 - prob_step_up

        trajectories_array_probs = (prob_step_up ** trajectories_array_no_up_steps
                                    * prob_step_down ** trajectories_array_no_down_steps)

        return trajectories_array_probs
        # 0. axis: all different trajectories
        # 1. axis: different probabilities to reach point (x_t, t) for respective trajectory


    @staticmethod
    def filter_trajectories_according_endpoint(trajectories_array_x: np.ndarray,
                                               trajectories_arrays: Iterable[np.ndarray, ...] = None):
        # initialization
        no_trajectories, T = np.shape(trajectories_array_x)

        list_trajectories_array_x = []
        nested_lists_trajectories_arrays = []

        if trajectories_arrays is not None:
            for array in trajectories_arrays:
                assert len(array) == no_trajectories, \
                    "axis of trajectories_array_x and arrays in trajectories_arrays corresponding to " \
                    "different trajectories must be equal"

                nested_lists_trajectories_arrays.append([])

        # filter trajectories
        for x in np.arange(- T, T + 1):
            trajectories_ending_at_x = trajectories_array_x[:, -1] == x
            list_trajectories_array_x.append(trajectories_array_x[trajectories_ending_at_x])

            if trajectories_arrays is not None:
                for j in len(trajectories_arrays):
                    nested_lists_trajectories_arrays[j].append(trajectories_arrays[j][trajectories_ending_at_x])

        if trajectories_arrays is not None:
            return list_trajectories_array_x, nested_lists_trajectories_arrays
        else:
            return list_trajectories_array_x
            # 0. index: different endpoints x_T = x, for x in np.arange(- T, T + 1)
            # 1. index/axis: all different trajectories with SAME endpoint x_T = x
            # 2. index/axis: x_t-values for respective trajectory


    # some exemplary observables for trajectories
    @staticmethod
    def calculate_moments_at_each_time(trajectories_array_x: np.ndarray, moment_up_to=2):
        pass


    @staticmethod
    def calculate_moments_integrated_over_time():
        pass


    @staticmethod
    def calculate_distance_to_t_axis():
        pass


    @staticmethod
    def calculate_difference_to_t_axis():
        pass


    @staticmethod
    def calculate_crossing_of_t_axis():
        pass





