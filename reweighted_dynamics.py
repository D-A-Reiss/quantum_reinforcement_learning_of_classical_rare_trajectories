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
from utilities import plot_prob_distribution


class ReweightedDynamics:
    def __init__(self, T: int, s: float, x_0=0., prob_step_up=0.5):
        # save inputs
        self.T = T
        self.s = s

        """
        # initialize arrays
        self.g_prime_array = np.empty((T + 1, 2 * T + 1))
        self.g_prime_array[:] = np.nan

        # calculate gauge transformation and reweighted dynamics
        self.calc_gauge_transformation_recursively(0, 0, T, s, x_0=x_0)
        for x in 2 * np.arange(T + 1) - T:
            self.calc_gauge_transformation_recursively(x, T, T, s, x_0=x_0)
        """

        self.g_array = self.calc_gauge_transformation_array(T, s, x_0=x_0, prob_step_up=prob_step_up)

        self.partition_function_Z = self.calc_partition_function(self.g_array, T)

        # plot_prob_distribution(T, self.g_array, set_title=False, title="$g$ ", plot_complement=True)

        self.P_W_array = self.calc_reweighted_dynamics(T, s, self.g_array, x_0=x_0, prob_step_up=prob_step_up)

        # plot reweighted dynamics
        plot_prob_distribution(T, self.P_W_array, set_title=False, title="$P_W$ ", plot_complement=True)



    @staticmethod
    def calc_weight_function(x: float, s: float, x_0=0.):
        """

        :param x:
        :param s:
        :return:
        """
        return np.exp(- s * (x - x_0)**2)


    @staticmethod
    def calc_partition_function(g_array: np.ndarray, T: int):
        return g_array[0, T]


    def calc_gauge_transformation_recursively(self, x: int, t: int, T: int, s: float, x_0=0.) -> float:
        """
        Calculates gauge transformation for balanced random walk bridges with softened constraint,
        called g' by Rose et al. ('21), by recursion
        (balanced: prob. to go 1 step up = prob. to go 1 step down;
        softened constraint: e^{- s x_T^2}) instead of \delta(x_T)).
        :param x: position
        :param t: time
        :param T: maximal time
        :param s: softening parameter
        :return: g'(x, t) for parameters T and s
        """
        assert t <= T, "t <= T required"

        if t == T:
            g_prime_value = 1

        elif t == T - 1:
            g_prime_value = self.calc_weight_function(x + 1, s, x_0=x_0) + self.calc_weight_function(x - 1, s, x_0=x_0)

        else:
            g_prime_value = (self.calc_gauge_transformation_recursively(x + 1, t + 1, T, s)
                             + self.calc_gauge_transformation_recursively(x - 1, t + 1, T, s))

        self.g_array[t, x + T] = g_prime_value

        return g_prime_value


    def calc_gauge_transformation_array(self, T: int, s: float, x_0=0., prob_step_up=0.5):
        # initialization
        g_array = np.empty((T + 1, 2 * T + 1))
        g_array[:] = np.nan

        def calc_g_value(x: int, t: int):
            if t == T:
                return 1

            elif t == T - 1:
                return prob_step_up * self.calc_weight_function(x + 1, s, x_0=x_0) \
                    + (1 - prob_step_up) * self.calc_weight_function(x - 1, s, x_0=x_0)

            else:
                return prob_step_up * g_array[t + 1, x + 1 + T] \
                    + (1 - prob_step_up) * g_array[t + 1, x - 1 + T]

        for t in np.arange(T + 1)[::-1]:
            for x in np.arange(- t, t + 1, 2):
                g_array[t, x + T] = calc_g_value(x, t)

        print(g_array)
        return g_array



    @staticmethod
    def calc_reweighted_dynamics(T: int, s: float, g_array: np.ndarray, x_0=0., prob_step_up=0.5):
        """
        Calculates reweighted dynamics, called P_W by Rose et al. ('21)
        :param T:
        :param s:
        :param g_array:
        :return:
        """
        x_values = np.arange(- T + 1, T)
        weights_array = np.ones((T, 2 * T - 1))
        #weights_array[T - 1, :] = np.exp(- s * (x_values + 1)**2)  # weights for up-step in last step
        weights_array[T - 1, :] = ReweightedDynamics.calc_weight_function(x_values + 1, s, x_0=x_0)

        P_W_array = g_array[1:, 2:] / g_array[:-1, 1:-1] * weights_array * prob_step_up
        # P_W_array only saves probabilities to go 1 step up (that to go 1 step down is given by normalization)

        return P_W_array




