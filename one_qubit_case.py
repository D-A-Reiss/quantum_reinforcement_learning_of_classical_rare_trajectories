from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import csv

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize


# CONSTANTS

sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., - 1j], [1j, 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])

v_x = 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
v_x_dagger = v_x


# FUNCTIONS
def r_n(alpha: float, n: str):
    assert n == "x" or n == "y" or n == "z", "n must be 'x' or 'y' or 'z'"

    if n == "x":
        sigma = sigma_x
    elif n == "y":
        sigma = sigma_y
    else:
        sigma = sigma_z

    return np.cos(alpha / 2) * np.identity(2) - 1j * np.sin(alpha / 2) * sigma


def to_binary_repr_list(num: int, bits: int):
    return [num // (2 ** j) % 2 for j in range(bits)][::-1]


def einsum_subscripts(*initial_subs: str, to=""):
    # function for better code readability:
    # converts the strings of initial subscripts in *args and the strings of final subscripts in **kwargs
    # (these strings can contain backslashes, numbers, primes, etc. like in SB21 and in my notes)
    # (e.g., one arg in *args might be arg="\vq',\lambda',\nu'" or arg="\vq_2',\nu_2'")
    # into one large einstein sum subscripts string for np.einsum
    list_letters = list(map(chr, range(97, 123)))  # list of all letters in the alphabet
    dict_subs = {}  # dictionary which contains to correspondences between subscripts in SB21/my notes
                         # and subscripts conforming with the conventions/requirements of np.einsum
    lists_initial_subs = [subs.split(",") for subs in initial_subs]
    list_final_subs = to.split(",")
    translated_subs = ""

    for subs_list in lists_initial_subs:
        for sub in subs_list:
            if sub not in dict_subs.keys():
                l = len(dict_subs)
                dict_subs.update({sub: list_letters[l]})
            translated_subs += dict_subs[sub]
        translated_subs += ","

    translated_subs = translated_subs[:-1]  # removes the last comma, which is too much
    translated_subs += "->"

    for sub in list_final_subs:
        translated_subs += dict_subs[sub]

    return translated_subs


def import_policy_from_csv(T: int, path='/Users/davidreiss/Desktop/Archiv/Quantum_PG_1/Plots/8_Final_policy_probabilities.csv'):
    # path to CSV file
    csv_file = path

    # create empty list to store data
    data_list_dict = []

    # open and read CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data_list_dict.append(row)

    # 'data' contains CSV data as a list of dictionaries
    # create empty list to store data
    data_array = np.empty((T, 2 * T - 1))
    data_array[:] = np.nan

    # convert to data structure used in class OptimalPolicyCalculations
    for row in data_list_dict:
        t = int(row["t"])
        x = int(row["x"])
        prob = float(row["probability of action_1"])

        data_array[t, x + T - 1] = prob

    return data_array



def fit_multivariate_func_leastsq(func: Callable[[np.ndarray, np.ndarray], np.ndarray], coords_array: np.ndarray,
                                  data_array: np.ndarray, params_initial_guess: np.ndarray, params_bounds=None,
                                  asserts=True, no_independent_vars: int = None):
    """
    Least-square fit of multivariate function func at coordinates in coords_array to data_array
    :param func:
    :param coords_array:
    :param data_array:
    :param params_initial_guess:
    :param params_bounds:
    :param asserts:
    :param no_independent_vars:
    :return: optimized_params, residual_mean_squared_error
    """
    # asserts
    if asserts:
        assert no_independent_vars is not None, \
            "If asserts == True, no_independent_vars must be specified!"

        shape_coords_array = np.shape(coords_array)
        shape_data_array = np.shape(data_array)

        assert shape_coords_array[-1] == no_independent_vars, \
            "Shape of coords_array[-1] must equal no_independent_vars!"
        assert shape_coords_array[:-1] == shape_data_array, \
            "Shape of coords_array[:-1] must equal shape of data_array!"

        func_vals_array = func(coords_array, params_initial_guess)

        shape_func_vals_array = np.shape(func_vals_array)

        assert shape_func_vals_array == shape_data_array, \
            "Shape of func(coords_array, params_initial_guess) must equal shape of data_array!"

    def cost_func(params):
        # compute values of func for coords_array and params
        func_vals_array = func(coords_array, params)

        # compute and return mean squared error as cost, ignoring NaNs
        return np.nanmean((func_vals_array - data_array) ** 2)

    result = minimize(cost_func, params_initial_guess, bounds=params_bounds)

    print("Fitting was successful:", result.success)

    optimized_params = result.x
    residual_mean_squared_error = result.fun

    return optimized_params, residual_mean_squared_error


# CLASSES
class FourierCoeffs:
    def __init__(self, no_layers: int):
        # choose all theta angles randomly
        theta_vector = 2 * np.pi * np.random.random(size=4 * no_layers)
        print(self.c_omega(theta_vector))

        # TODO: implement formula for Fourier coefficients with variable no. of data uploading layers and
        #  check whether for several combinations of random angles/parameters they are non-zero
        # TODO: think about how to show the result of this numerical experiment in general/analytically or
        #  how to argue for the result


    def calc_a_kj_1_qubit(self, k_vector: np.ndarray, j_vector: np.ndarray, theta_vector: np.ndarray):
        """

        :param k_vector:
        :param j_vector:
        :param theta_vector:
        :return:
        """
        no_thetas = len(theta_vector)

        assert no_thetas % 4 == 0, "# thetas must be a multiple of 4"
        assert len(k_vector) == len(j_vector), "len(k_vector) == len(j_vector) required"
        assert len(k_vector) == 7 + (no_thetas // 4 - 1) * 8, " len(k_vector) == 7 + (no_thetas // 4 - 1) * 8 required"

        # case of only one data-uploading layer
        if no_thetas // 4 == 1:
            # first half of product: unitary transformation
            result = (r_n(theta_vector[2], "y")[:, :]
                      @ v_x[:, j_vector[1]]
                      # = vector
                      * v_x_dagger[j_vector[1], :]
                      @ r_n(theta_vector[1], "z")[:, :]
                      @ r_n(theta_vector[0], "y")[:, :]
                      @ v_x[:, j_vector[0]]
                      # = scalar
                      * v_x_dagger[j_vector[0], 0]
                      # = scalar
                      )

            # observable
            result = (r_n(- theta_vector[3], "z").transpose()[:, :]
                      # TODO: check whether previous line is correct
                      @ sigma_z
                      @ r_n(theta_vector[3], "z")[:, :]
                      @ result)

            # second half of product: conjugate transpose of unitary transformation
            result *= (r_n(- theta_vector[2], "y").transpose()[k_vector[6], k_vector[5]] *
                       v_x.transpose()[k_vector[5], k_vector[4]] *
                       v_x_dagger.transpose()[k_vector[4], k_vector[3]] *
                       r_n(- theta_vector[1], "z").transpose()[k_vector[3], k_vector[2]] *
                       r_n(- theta_vector[0], "y").transpose()[k_vector[2], k_vector[1]] *
                       v_x.transpose()[k_vector[1], k_vector[0]] *
                       v_x_dagger.transpose()[k_vector[0], 0])
            return result

        # case of at least two/more than one data-uploading layers
        # code implements terms of expectation value from right to left
        # first half of product: unitary transformation
        result = (r_n(theta_vector[3], "z")[j_vector[7], j_vector[6]] *
                  r_n(theta_vector[2], "y")[j_vector[6], j_vector[5]] *
                                        v_x[j_vector[5], j_vector[4]] *
                                 v_x_dagger[j_vector[4], j_vector[3]] *
                  r_n(theta_vector[1], "z")[j_vector[3], j_vector[2]] *
                  r_n(theta_vector[0], "y")[j_vector[2], j_vector[1]] *
                                        v_x[j_vector[1], j_vector[0]] *
                                 v_x_dagger[j_vector[0], 0])

        for i in range(no_thetas // 4 - 2):
            s = 7 + i * 8  # shift/offset due to 1st and i-many more previous layers
            s_theta = 4 + i * 4

            result *= (r_n(theta_vector[s_theta + 3], "z")[j_vector[s + 8], j_vector[s + 7]] *
                       r_n(theta_vector[s_theta + 2], "y")[j_vector[s + 7], j_vector[s + 6]] *
                                                       v_x[j_vector[s + 6], j_vector[s + 5]] *
                                                v_x_dagger[j_vector[s + 5], j_vector[s + 4]] *
                       r_n(theta_vector[s_theta + 1], "z")[j_vector[s + 4], j_vector[s + 3]] *
                       r_n(theta_vector[s_theta + 0], "y")[j_vector[s + 3], j_vector[s + 2]] *
                                                       v_x[j_vector[s + 2], j_vector[s + 1]] *
                                                v_x_dagger[j_vector[s + 1], j_vector[s + 0]])

        s = 7 + (no_thetas // 4 - 2) * 8
        s_theta = 4 + (no_thetas // 4 - 2) * 4

        result *= (r_n(theta_vector[s_theta + 2], "y")[j_vector[s + 7], j_vector[s + 6]] *
                                                   v_x[j_vector[s + 6], j_vector[s + 5]] *
                                            v_x_dagger[j_vector[s + 5], j_vector[s + 4]] *
                   r_n(theta_vector[s_theta + 1], "z")[j_vector[s + 4], j_vector[s + 3]] *
                   r_n(theta_vector[s_theta + 0], "y")[j_vector[s + 3], j_vector[s + 2]] *
                                                   v_x[j_vector[s + 2], j_vector[s + 1]] *
                                            v_x_dagger[j_vector[s + 1], j_vector[s + 0]])

        # observable
        result *= (r_n(- theta_vector[s_theta + 3], "z").transpose()[k_vector[s + 7], :]
                   # TODO: check whether previous line is correct
                   @ sigma_z
                   @ r_n(theta_vector[s_theta + 3], "z")[:, j_vector[s + 7]])

        # second half of product: conjugate transpose of unitary transformation
        result *= (r_n(- theta_vector[s_theta + 2], "y").transpose()[k_vector[s + 7], k_vector[s + 6]] *
                                                     v_x.transpose()[k_vector[s + 6], k_vector[s + 5]] *
                                              v_x_dagger.transpose()[k_vector[s + 5], k_vector[s + 4]] *
                   r_n(- theta_vector[s_theta + 1], "z").transpose()[k_vector[s + 4], k_vector[s + 3]] *
                   r_n(- theta_vector[s_theta + 0], "y").transpose()[k_vector[s + 3], k_vector[s + 2]] *
                                                     v_x.transpose()[k_vector[s + 2], k_vector[s + 1]] *
                                              v_x_dagger.transpose()[k_vector[s + 1], k_vector[s + 0]])

        for i in range(no_thetas // 4 - 2):
            s = 7 + i * 8  # shift/offset due to 1st and i-many more previous layers
            s_theta = 4 + i * 4

            result *= (r_n(- theta_vector[s_theta + 3], "z").transpose()[k_vector[s + 8], k_vector[s + 7]] *
                       r_n(- theta_vector[s_theta + 2], "y").transpose()[k_vector[s + 7], k_vector[s + 6]] *
                                                         v_x.transpose()[k_vector[s + 6], k_vector[s + 5]] *
                                                  v_x_dagger.transpose()[k_vector[s + 5], k_vector[s + 4]] *
                       r_n(- theta_vector[s_theta + 1], "z").transpose()[k_vector[s + 4], k_vector[s + 3]] *
                       r_n(- theta_vector[s_theta + 0], "y").transpose()[k_vector[s + 3], k_vector[s + 2]] *
                                                         v_x.transpose()[k_vector[s + 2], k_vector[s + 1]] *
                                                  v_x_dagger.transpose()[k_vector[s + 1], k_vector[s + 0]])

        result *= (r_n(- theta_vector[3], "z").transpose()[k_vector[7], k_vector[6]] *
                   r_n(- theta_vector[2], "y").transpose()[k_vector[6], k_vector[5]] *
                                           v_x.transpose()[k_vector[5], k_vector[4]] *
                                    v_x_dagger.transpose()[k_vector[4], k_vector[3]] *
                   r_n(- theta_vector[1], "z").transpose()[k_vector[3], k_vector[2]] *
                   r_n(- theta_vector[0], "y").transpose()[k_vector[2], k_vector[1]] *
                                           v_x.transpose()[k_vector[1], k_vector[0]] *
                                    v_x_dagger.transpose()[k_vector[0], 0])

        return result


    def c_omega(self, theta_vector: np.ndarray):
        """

        :param theta_vector:
        :return:
        """

        # FIXME: according to corrections regarding k- and j-vectors in calc_a_kj_1_qubit
        # TODO: take care of the fact that in our case each coefficient c_omega is labelled by 2 frequencies omega

        no_thetas = len(theta_vector)

        len_kj_vectors = 7 + (no_thetas // 4 - 1) * 8
        # max_omega = len_kj_vectors

        omega_array = np.arange(- len_kj_vectors, len_kj_vectors + 1)
        c_omega_array = np.zeros(2 * len_kj_vectors + 1)
        # initializes array of c_omega-values

        all_kj_vectors = np.array([to_binary_repr_list(i, len_kj_vectors) for i in range(2 ** len_kj_vectors)])
        # generates all possible k- and j-vectors

        """
        lambda_kj_vectors = (-1) ** all_kj_vectors
        # calculates all corresponding eigenvalues \lambda_{j,k = 0} = 1 and \lambda_{j,k = 1} = -1

        sum_lambda_kj_vectors = np.sum(lambda_kj_vectors, axis=1)
        # denoted as \Lambda_{k,j} by Schuld, Sweke, and Meyer

        sum_lambda_k_mesh, sum_lambda_j_mesh = np.meshgrid(sum_lambda_kj_vectors, sum_lambda_kj_vectors,
                                                           indexing="ij")

        diff_sums_lambdas = sum_lambda_k_mesh - sum_lambda_j_mesh
        """

        for k_vector in all_kj_vectors:
            sum_lambda_k = np.sum((-1) ** k_vector)

            for j_vector in all_kj_vectors:
                sum_lambda_j = np.sum((-1) ** j_vector)

                diff_sums_lambda = sum_lambda_k - sum_lambda_j

                a_kj_1_qubit = self.calc_a_kj_1_qubit(k_vector, j_vector, theta_vector)

                c_omega_array = np.add(c_omega_array, a_kj_1_qubit,
                                       where=(omega_array == diff_sums_lambda))

        return c_omega_array


class OptimalPolicyCalculations:
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
        self.plot_prob_distribution(T, self.P_W_array, "$P_W$ ")


        ### plot policy (in the 1-qubit case w/o data RE-uploading)
        ### as SANITY CHECK also in terms of variational parameters \theta_1, \theta_2, and \theta_3)
        # import results of QRL algorithm
        self.policy_array = import_policy_from_csv(T, path='/Users/davidreiss/Desktop/Archiv/Quantum_PG_2/Plots/8_Final_policy_probabilities.csv')
        self.plot_mask = np.isnan(self.policy_array)
        params_1_qubit_array = np.array([1.003468, 1.0284932]
                                        + [0.8490333, 1.8261642, 1.0306203]#, 0.94512033]
                                        + [2.0050492])#, -2.005049])
        # input scalings, variational params, output scaling

        # calculate amplitudes and phases from variational parameters
        params_array = self.calc_params_array_1_qubit_case(params_1_qubit_array)

        # plot policy and prediction from analytical calculation
        self.plot_prob_distribution(T, self.policy_array, "$\pi$ ")
        self.plot_prob_distribution(T, self.softmax_policy(self.coords_array, params_array), "$P_\\theta$ ", masking=True)


        ### fits to reweighted dynamics
        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        """
        optimized_params_1, residual_mean_squared_error_1 = \
            fit_multivariate_func_leastsq(self.softmax_policy, self.coords_array, self.P_W_array,
                                          params_initial_guess=np.ones(2 + 2 * no_pos_freqs * no_freqs + 1),
                                          params_bounds=([(-np.inf, np.inf)] * (2 + no_pos_freqs * no_freqs)
                                                         + [(0., 2 * np.pi)] * (no_pos_freqs * no_freqs)
                                                         + [(-np.inf, np.inf)]),
                                          no_independent_vars=2)
        optimized_params_2, residual_mean_squared_error_2 = \
            fit_multivariate_func_leastsq(self.softmax_policy, self.coords_array, self.policy_array,
                                          params_initial_guess=np.ones(2 + 2 * no_pos_freqs * no_freqs + 1),
                                          params_bounds=([(-np.inf, np.inf)] * (2 + no_pos_freqs * no_freqs)
                                                         + [(0., 2 * np.pi)] * (no_pos_freqs * no_freqs)
                                                         + [(-np.inf, np.inf)]),
                                          no_independent_vars=2)
        """
        optimized_params_3, residual_mean_squared_error_3 = \
            fit_multivariate_func_leastsq(self.softmax_policy_thetas, self.coords_array, self.P_W_array,
                                          params_initial_guess=np.ones(2 + 3 + 1),
                                          params_bounds=([(-np.inf, np.inf)] * 2
                                                         + [(0., 2 * np.pi)] * 3
                                                         + [(-np.inf, np.inf)]),
                                          no_independent_vars=2)
        optimized_params_4, residual_mean_squared_error_4 = \
            fit_multivariate_func_leastsq(self.softmax_policy_thetas, self.coords_array, self.policy_array,
                                          params_initial_guess=np.ones(2 + 3 + 1),
                                          params_bounds=([(-np.inf, np.inf)] * 2
                                                         + [(0., 2 * np.pi)] * 3
                                                         + [(-np.inf, np.inf)]),
                                          no_independent_vars=2)

        print(optimized_params_3, optimized_params_4)

        """
        vals_fitted_func_array_1 = self.softmax_policy(self.coords_array, optimized_params_1)
        vals_fitted_func_array_2 = self.softmax_policy(self.coords_array, optimized_params_2)
        """
        vals_fitted_func_array_3 = self.softmax_policy_thetas(self.coords_array, optimized_params_3)
        vals_fitted_func_array_4 = self.softmax_policy_thetas(self.coords_array, optimized_params_4)

        #self.plot_prob_distribution(T, vals_fitted_func_array_1, masking=True)
        #self.plot_prob_distribution(T, vals_fitted_func_array_2, masking=True)
        self.plot_prob_distribution(T, vals_fitted_func_array_3, masking=True)
        self.plot_prob_distribution(T, vals_fitted_func_array_4, masking=True)

        #self.plot_prob_distribution(T, np.abs(vals_fitted_func_array - self.P_W_array))
        #self.plot_prob_distribution(T, np.abs(vals_fitted_func_array_2 - self.policy_array))

        """
        print(residual_mean_squared_error, residual_mean_squared_error_2)
        """


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
            g_prime_value = np.exp(- s * (x + 1)**2) + np.exp(- s * (x - 1)**2)

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


    def plot_prob_distribution(self, T: int, prob_array: np.ndarray, title="", masking=False):
        """
        Calculates reweighted dynamics as function of t and x
        :param title:
        :param T:
        :param prob_array:
        :return:
        """
        if masking:
            prob_array = np.where(self.plot_mask, np.nan, prob_array)
        prob_array = np.swapaxes(prob_array, 0, 1)
        # now indices x,t
        prob_array = prob_array[::-1, :]

        # plot as heat map like in paper draft
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(prob_array, cmap='viridis')

        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.set_title(title + "(to go 1 step up)")
        ax.set_xlabel("time $t$")
        ax.set_ylabel("position $x$")

        ax.set_xticks(np.array([0, 1 * T//4, 2 * T//4, 3 * T//4, T]),
                      labels=[str(0), str(1 * T//4), str(2 * T//4), str(3 * T//4), str(T)])

        ax.set_yticks(np.array([0, T - 1, 2 * T - 2]),
                      labels=[str(T - 1), str(0), str(1 - T)])

        # fig.savefig("figure.pdf")

        plt.show()

        """
        # plot as weighted graph
        # to this end use
        # https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx
        # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_array.html

        # construct adjacency matrix
        no_t, no_x = np.shape(P_W_array)
        no_vertices = (no_t + 1) * (no_x + 2)
        adjacency_matrix = np.zeros((no_vertices, no_vertices))

        for t in range(no_t):
            for x in range(no_x):
                vertex = t * no_x + x
                vertex_step_up = (t + 1) * no_x + (x + 1)
                vertex_step_down = (t + 1) * no_x + (x - 1)

                adjacency_matrix[vertex, vertex_step_up] = adjacency_matrix[vertex_step_up, vertex] = \
                    P_W_array[t, x]
                adjacency_matrix[vertex, vertex_step_down] = adjacency_matrix[vertex_step_down, vertex] = \
                    (1 - P_W_array[t, x])

                graph = nx.from_numpy_array(adjacency_matrix)
                nx.draw(graph)
                plt.show()
        """


    def calc_Fourier_coeffs_1_qubit_case(self):
        pass


    def calc_params_array_1_qubit_case(self, params_1_qubit_array: np.ndarray, in_terms_of_thetas=True):
        """

        :param params_1_qubit_array: contains lambdas, either thetas or INDEPENDENT amplitudes and phases, and w
        :return:
        """
        # asserts
        if in_terms_of_thetas:
            thetas_array = params_1_qubit_array[2:-1]
            assert len(thetas_array) == 3, \
                "amplitudes and phases in truncated Fourier series are determined in the 1-qubit case " \
                "by 3 angles theta_1, theta_2, and theta_3, which must be supplied in params_1_qubit_array[2:-1]"
        else:
            assert len(params_1_qubit_array[2:-1]) == 6, \
                "amplitudes and phases in truncated Fourier series are determined in the 1-qubit case " \
                "by 3 non-zero amplitudes and 3 non-zero phases, which must be supplied in this order in " \
                "params_1_qubit_array[2:-1]"

        if in_terms_of_thetas:
            theta_1 = thetas_array[0]
            theta_2 = thetas_array[1]
            theta_3 = thetas_array[2]

            # complex-valued coefficients of truncated Fourier series
            c_array = np.zeros(2 * 3, dtype=complex)
            c_array[1 * 3 + 0] = 1 / 4 * (np.cos(theta_1) - np.cos(theta_2) - 1j * np.sin(theta_1) * np.sin(theta_2)) \
                                 * np.cos(theta_3)
            c_array[1 * 3 + 1] = - 1 / 2 * (np.sin(theta_1) * np.cos(theta_2) - 1j * np.sin(theta_2)) \
                                 * np.sin(theta_3)
            c_array[1 * 3 + 2] = 1 / 4 * (np.cos(theta_1) + np.cos(theta_2) - 1j * np.sin(theta_1) * np.sin(theta_2)) \
                                 * np.cos(theta_3)

            # corresponding amplitudes and phases in truncated Fourier series
            a_array = 2 * np.abs(c_array)
            phi_array = np.angle(c_array)
        else:
            a_array = np.zeros(2 * 3)
            phi_array = np.zeros(2 * 2)

            for i in range(3):
                a_array[1 * 3 + i] = params_1_qubit_array[2 + i]
                phi_array[1 * 2 + i] = params_1_qubit_array[2 + 3 + i]

        # construct params_array suitable for function softmax_policy
        params_array = np.zeros(2 + len(a_array) + len(phi_array) + 1)
        params_array[:2] = params_1_qubit_array[:2]
        params_array[2:-1] = np.concatenate((a_array, phi_array))
        params_array[-1:] = params_1_qubit_array[-1:]

        return params_array


    def softmax_policy(self, coords_array: np.ndarray, params_array: np.ndarray):
        # split params_array in "subarrays"
        lambda_array = params_array[:2]
        coeffs_array = params_array[2:-1]
        w = params_array[-1:]

        no_pos_freqs = self.no_layers + 1
        no_freqs = 2 * self.no_layers + 1

        # asserts
        if self.asserts:
            assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
                "length of params_array not correct; it must contain 2 input scaling parameters, " \
                "(self.no_layers + 1) * (2 * self.no_layers + 1) amplitudes, " \
                "(self.no_layers + 1) * (2 * self.no_layers + 1) phases, " \
                "and 1 output scaling parameter"
            # TODO: implement further asserts

        # computations
        g_t = np.arctan(lambda_array[0] * coords_array[..., 0])
        g_x = np.arctan(lambda_array[1] * coords_array[..., 1])

        pos_freqs = np.arange(0, self.no_layers + 1)
        freqs = np.arange(- self.no_layers, self.no_layers + 1)
        freqs_t, freqs_x = np.meshgrid(pos_freqs, freqs, indexing="ij")
        # -> increasing first index corresponds to increasing frequency for oscillations in t,
        #    increasing second ----------------------------"---------------------------- in x
        # NOTE: due to the symmetry cos(-x) = cos(x), one does NOT have to consider negative frequencies for
        # either g_t or g_x; here g_t is chosen

        amplitudes = np.broadcast_to(coeffs_array[:no_pos_freqs * no_freqs].reshape(no_pos_freqs, no_freqs),
                                     (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))
        phases = np.broadcast_to(coeffs_array[no_pos_freqs * no_freqs:].reshape(no_pos_freqs, no_freqs),
                                 (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))

        avg_Z = amplitudes * np.cos(np.einsum(einsum_subscripts("ft,fx",
                                                                "gt,gx",
                                                                to="gt,gx,ft,fx"),
                                              freqs_t,
                                              g_t)
                                    + np.einsum(einsum_subscripts("ft,fx",
                                                                  "gt,gx",
                                                                  to="gt,gx,ft,fx"),
                                                freqs_x,
                                                g_x)
                                    + phases)

        return 1 / (np.exp(w * np.einsum(einsum_subscripts("gt,gx,ft,fx",
                                                           to="gt,gx"),
                                         avg_Z))
                    + 1)
        # TODO: check whether this last form is correct!


    def softmax_policy_thetas(self, coords_array: np.ndarray, params_1_qubit_array: np.ndarray):
        return self.softmax_policy(coords_array, self.calc_params_array_1_qubit_case(params_1_qubit_array))


    """
    def fit_softmax_policy_to_data(self, T: int, policy_array: np.ndarray, no_uploading_layers=1):
        def input_scaling(y: int, lambda_y: float):
            return np.arctan(lambda_y * y)

        def softmax_policy(t_array: np.ndarray, x_array: np.ndarray, lambda_array: np.ndarray, theta_array: np.ndarray,
                           w: float, no_uploading_layers=1):
            g_t = input_scaling(t_array, lambda_array[0])
            g_x = input_scaling(x_array, lambda_array[1])
            g_t, g_x = np.meshgrid(g_t, g_x, indexing="ij")

            freqs = np.arange(- no_uploading_layers, no_uploading_layers + 1)
            no_freqs = len(freqs)
            freqs_t, freqs_x = np.meshgrid(freqs, freqs, indexing="ij")

            amplitudes = np.broadcast_to(theta_array[:no_freqs ** 2].reshape(no_freqs, no_freqs),
                                         (len(t_array), len(x_array), no_freqs, no_freqs))
            phases = np.broadcast_to(theta_array[no_freqs ** 2:].reshape(no_freqs, no_freqs),
                                     (len(t_array), len(x_array), no_freqs, no_freqs))

            avg_Z = amplitudes * np.cos(np.einsum(einsum_subscripts("ft,fx",
                                                                    "gt,gx",
                                                                    to="gt,gx,ft,fx"),
                                                  freqs_t,
                                                  g_t)
                                        + np.einsum(einsum_subscripts("ft,fx",
                                                                      "gt,gx",
                                                                      to="gt,gx,ft,fx"),
                                                    freqs_x,
                                                    g_x)
                                        + phases)

            return 1 / (np.exp(w * np.einsum(einsum_subscripts("gt,gx,ft,fx",
                                                               to="gt,gx"),
                                             avg_Z))
                        + 1)

        def fit_function(tx_array: np.ndarray, fit_params: list):
            t_array = np.unique(tx_array[:, 0])
            x_array = np.unique(tx_array[:, 1])

            lambda_array = np.array(fit_params[:2])
            theta_array = np.array(fit_params[2:-1])
            w = fit_params[-1]

            return softmax_policy(t_array, x_array, lambda_array, theta_array, w,
                                  no_uploading_layers=no_uploading_layers)

        t_values = np.arange(T)
        x_values = np.arange(- T + 1, T)

        tx_array = np.array([(t, x) for t in t_values for x in x_values])

        policy_array = policy_array.ravel()
        # to ensure that policy_array is 1D array with same number of elements as data points in tx_array

        print(np.shape(policy_array), np.shape(tx_array))
        print(np.shape(fit_function(tx_array, [1] * (2 + 2 * (2 * no_uploading_layers + 1) ** 2 + 1))))

        return curve_fit(fit_function, tx_array, policy_array, check_finite=False, nan_policy="omit", full_output=True,
                         p0=([1] * (2 + 2 * (2 * no_uploading_layers + 1) + 1)),
                         # because there are 2 input weights/lambdas, 2 * no_uploading_layers + 1 amplitudes,
                         # 2 * no_uploading_layers + 1 phases, and 1 output weight w
                         bounds=([-np.inf, -np.inf] + [0] * 2 * (2 * no_uploading_layers + 1) + [-np.inf],
                                 [np.inf, np.inf] + [2 * np.pi] * 2 * (2 * no_uploading_layers + 1) + [np.inf])
                         )

        # use curve_fit with bounds and full_output=True, compute residual sum of squares from key fvec of returned infodict
        """