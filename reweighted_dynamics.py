# IMPORTS #####
import numpy as np
from utilities import plot_prob_distribution


# ROUTINES ######
class ReweightedDynamics:
    def __init__(self, T: int, s: float, no_layers: int):
        self.asserts = True
        self.no_layers = no_layers

        # initialize arrays
        self.g_prime_array = np.empty((T + 1, 2 * T + 1))
        self.g_prime_array[:] = np.nan

        self.t_values = np.arange(T)
        self.x_values = np.arange(- T + 1, T)

        self.coords_array = np.array([[(t, x) for x in self.x_values]
                                      for t in self.t_values])

        # calculate gauge transformation and reweighted dynamics
        self.calc_gauge_transformation(0, 0, T, s)
        for x in 2 * np.arange(T + 1) - T:
            self.calc_gauge_transformation(x, T, T, s)

        self.P_W_array = self.calc_reweighted_dynamics(T, s, self.g_prime_array)

        # plot reweighted dynamics
        plot_prob_distribution(T, self.P_W_array, set_title=False, title="$P_W$ ")

        # calculate and plot value functions for original and reweighted dynamics
        self.P_array = np.where(np.isnan(self.P_W_array), np.nan, 1/2)

        self.value_func_array = np.empty((T + 1, 2 * T + 1))
        self.value_func_array[:] = np.nan
        self.calc_value_function(0, 0, T, s, self.P_array, self.P_array)

        self.plot_prob_distribution(T, np.log(-self.value_func_array), set_title=False, title="$V_{P}$ ", diff=True)

        self.value_func_array = np.empty((T + 1, 2 * T + 1))
        self.value_func_array[:] = np.nan
        self.calc_value_function(0, 0, T, s, self.P_W_array, self.P_array)

        self.plot_prob_distribution(T, np.log(-self.value_func_array), set_title=False, title="$V_{P_W}$ ", diff=True)


    def calc_weight_function(self, x: float, s: float):
        return np.exp(- s * x**2)


    def calc_gauge_transformation(self, x: int, t: int, T: int, s: float) -> float:
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
            g_prime_value = self.calc_weight_function(x + 1, s) + self.calc_weight_function(x - 1, s)

        else:
            g_prime_value = (self.calc_gauge_transformation(x + 1, t + 1, T, s)
                             + self.calc_gauge_transformation(x - 1, t + 1, T, s))

        self.g_prime_array[t, x + T] = g_prime_value

        return g_prime_value


    def calc_reweighted_dynamics(self, T: int, s: float, g_prime_array: np.ndarray):
        """
        Calculates reweighted dynamics, called P_W by Rose et al. ('21)
        :return:
        """
        weights_array = np.ones((T, 2 * T - 1))
        weights_array[T - 1, :] = np.exp(- s * (self.x_values + 1)**2)  # weights for up-step in last step

        P_W_array = g_prime_array[1:, 2:] / g_prime_array[:-1, 1:-1] * weights_array
        # P_W_array only saves probabilities to go 1 step up (that to go 1 step down is given by normalization)

        return P_W_array


    def calc_reward(self, x_t: int, x_prev_t: int, t: int, T: int, s: float,
                    p_theta_distribution: np.ndarray, p_distribution: np.ndarray):
        if t == T:
            weight = self.calc_weight_function(x_t, s)
        else:
            weight = 1

        p_theta = p_theta_distribution[t - 1, x_prev_t + T - 1]
        p = p_distribution[t - 1, x_prev_t + T - 1]

        if x_t == x_prev_t + 1:
            return np.log(weight) - np.log(p_theta) + np.log(p)
        elif x_t == x_prev_t - 1:
            return np.log(weight) - np.log(1 - p_theta) + np.log(1 - p)


    def calc_value_function(self, x: int, t: int, T: int, s: float,
                            p_theta_distribution: np.ndarray, p_distribution: np.ndarray):
        # Bellman eq. adapted to our random walk
        if t == T:
            value_func_val = 0.
        else:
            p_theta = p_theta_distribution[t, x + T - 1]
            # regarding the indices consider the simplest example T = 2:
            # np.shape(p_theta_distribution) = (2, 3) corresponding to values t = 0, 1, and x = -1, 0, 1

            if p_theta == 0.:
                value_func_val = (1 - p_theta) \
                                 * (self.calc_value_function(x - 1, t + 1, T, s, p_theta_distribution, p_distribution)
                                    + self.calc_reward(x - 1, x, t + 1, T, s, p_theta_distribution, p_distribution))
            elif p_theta == 1.:
                value_func_val = p_theta \
                                 * (self.calc_value_function(x + 1, t + 1, T, s, p_theta_distribution, p_distribution)
                                    + self.calc_reward(x + 1, x, t + 1, T, s, p_theta_distribution, p_distribution))
            else:
                value_func_val = p_theta \
                                 * (self.calc_value_function(x + 1, t + 1, T, s, p_theta_distribution, p_distribution)
                                    + self.calc_reward(x + 1, x, t + 1, T, s, p_theta_distribution, p_distribution)) \
                                 + (1 - p_theta) \
                                 * (self.calc_value_function(x - 1, t + 1, T, s, p_theta_distribution, p_distribution)
                                    + self.calc_reward(x - 1, x, t + 1, T, s, p_theta_distribution, p_distribution))

        self.value_func_array[t, x + T - 1] = value_func_val

        return value_func_val

