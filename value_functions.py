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
from reweighted_dynamics import ReweightedDynamics
from utilities import ConsistentParametersClass


class ValueFunction(ConsistentParametersClass):
    def __init__(self, T: int, s: float, p_theta_distribution: np.ndarray):
        super().__init__()

        # save inputs
        self.T = T
        self.s = s
        self.p_theta_distribution = p_theta_distribution

        # initialization
        self.p_distribution = np.where(np.isnan(p_theta_distribution), np.nan, 1 / 2)

        self.value_func_array = np.empty((T + 1, 2 * T + 1))
        self.value_func_array[:] = np.nan

        # compute value function
        self.calc_value_function(0, 0, T, s, p_theta_distribution, self.p_distribution)


    @property
    def all_init_params_dict(self):
        return {"T": self.T, "s": self.s}


    @staticmethod
    def calc_reward(action_a: int, position_x: int, time_t: int, tot_time: int, s: float,
                    p_theta_distribution: np.ndarray, p_distribution: np.ndarray) -> float:
        """
        Calculate reward for transition time_t - 1, position_x --> time_t, position_x + action_a according to
        reinforcement learning framework applied to 1D random walk.

        Parameters:
            action_a: action_a of random walker (either -1 or +1, representing left/down or right/up step)
            position_x: position of random walker at time_t - 1
            time_t: time step
            tot_time: total number of time steps
            s: reward function parameter (balancing closeness of p_theta_distribution to p_distribution vs.
                                          probability of generating specific rare trajectories)
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + tot_time - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            p_distribution: distribution P of original transition probabilities
                            (p_distribution[t - 1, position_x + tot_time - 1] is expected to be probability
                            for transition time_t - 1, position_x --> time_t, position_x + 1 according to P)

        Returns:
            reward for transition time_t - 1, position_x --> time_t, position_x + action_a
        """
        
        assert action_a in [-1, 1], "action_a must be either -1 or +1"

        # calculate weight 
        if time_t == tot_time:
            weight = ReweightedDynamics.calc_weight_function(position_x + action_a, s)
        else:
            weight = 1

        # probabilities for up/right step
        p_theta = p_theta_distribution[time_t - 1, position_x + tot_time - 1]
        p = p_distribution[time_t - 1, position_x + tot_time - 1]
        # regarding the indices consider the simplest example tot_time = 2:
        # np.shape(p_theta_distribution) = (2, 3) corresponding to values t = 0, 1, and x = -1, 0, 1

        # calculate reward
        if action_a == 1:
            return np.log(weight) - np.log(p_theta) + np.log(p)
        else:
            return np.log(weight) - np.log(1 - p_theta) + np.log(1 - p)
        

    def calc_value_function(self, position_x: int, time_t: int, tot_time: int, s: float,
                            p_theta_distribution: np.ndarray, p_distribution: np.ndarray) -> float:
        """
        Calculate value function for random walk at position_x and time_t via Bellman equation recursively and
        store results in self.value_func_array.
        The value function is defined as the expected sum of rewards for all future transitions starting from
        position_x and time_t up to tot_time.

        Parameters:
            position_x: position
            time_t: time
            tot_time: total number of time steps
            s: reward function parameter (balancing closeness of p_theta_distribution to p_distribution vs.
                                          probability of generating specific rare trajectories)
            p_theta_distribution: distribution P_theta of arbitrary transition probabilities
                                  (p_theta_distribution[t - 1, position_x + tot_time - 1] is expected to be probability
                                  for transition time_t - 1, position_x --> time_t, position_x + 1 according to P_theta)
            p_distribution: distribution P of original transition probabilities
                            (p_distribution[t - 1, position_x + tot_time - 1] is expected to be probability
                            for transition time_t - 1, position_x --> time_t, position_x + 1 according to P)

        Returns:
            value function for random walk at position_x and time_t
        """

        # calculate value function value via Bellman eq.
        if time_t == tot_time:
            value_func_val = 0.  # boundary condition

        else:
            # probability for up/right step
            p_theta = p_theta_distribution[time_t, position_x + tot_time - 1]
            # regarding the indices consider the simplest example tot_time = 2:
            # np.shape(p_theta_distribution) = (2, 3) corresponding to values t = 0, 1, and x = -1, 0, 1

            if p_theta == 0.:
                value_func_val = (1 - p_theta) \
                                 * (self.calc_value_function(position_x - 1, time_t + 1, tot_time, s,
                                                             p_theta_distribution, p_distribution)
                                    + self.calc_reward(-1, position_x, time_t + 1, tot_time, s,
                                                       p_theta_distribution, p_distribution))

            elif p_theta == 1.:
                value_func_val = p_theta \
                                 * (self.calc_value_function(position_x + 1, time_t + 1, tot_time, s,
                                                             p_theta_distribution, p_distribution)
                                    + self.calc_reward(1, position_x, time_t + 1, tot_time, s,
                                                       p_theta_distribution, p_distribution))

            else:
                value_func_val = p_theta \
                                 * (self.calc_value_function(position_x + 1, time_t + 1, tot_time, s,
                                                             p_theta_distribution, p_distribution)
                                    + self.calc_reward(1, position_x, time_t + 1, tot_time, s,
                                                       p_theta_distribution, p_distribution)) \
                                 + (1 - p_theta) \
                                 * (self.calc_value_function(position_x - 1, time_t + 1, tot_time, s,
                                                             p_theta_distribution, p_distribution)
                                    + self.calc_reward(-1, position_x, time_t + 1, tot_time, s,
                                                       p_theta_distribution, p_distribution))

        # store value function value
        self.value_func_array[time_t, position_x + tot_time - 1] = value_func_val

        return value_func_val