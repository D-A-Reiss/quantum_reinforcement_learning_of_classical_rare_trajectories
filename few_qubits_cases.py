# IMPORTS #####
import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import orqviz

from typing import Callable
from scipy.optimize import minimize

from reweighted_dynamics import ReweightedDynamics
from utilities import import_policy_from_csv, einsum_subscripts, ProgressBar, plot_prob_distribution


# CONSTANTS #####
# global settings
global_size = 18
font_size = 18

mpl.rcParams['font.serif'] = 'cmr10'  # alternative: 'Times New Roman'
mpl.rcParams['font.sans-serif'] = 'cmr10'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams["axes.labelsize"] = global_size
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = global_size
mpl.rcParams["ytick.labelsize"] = global_size
plt.rcParams.update({"font.size": font_size})

sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., - 1j], [1j, 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])

sp_identity = sp.eye(2)
sp_sigma_x = sp.Matrix([[0, 1], [1, 0]])
sp_sigma_y = sp.Matrix([[0, - 1j], [1j, 0]])
sp_sigma_z = sp.Matrix([[1, 0], [0, -1]])

v_x = 1 / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
v_x_dagger = v_x


# ROUTINES #####
def r_n(alpha: float | sp.Symbol, n: str, sympy_expr=False) -> np.ndarray:
    """
    Compute rotations on Bloch sphere, if sympy_expr == True the symbolic (SymPy) expressions, else the numerical ones.

    Parameters:
        alpha: rotation angle
        n: rotation axis, either "x" or "y" or "z"
        sympy_expr: see description above

    Returns:
        np.ndarray
    """
    identity = np.identity(2) if not sympy_expr else sp_identity

    if n == "x":
        sigma = sigma_x if not sympy_expr else sp_sigma_x
    elif n == "y":
        sigma = sigma_y if not sympy_expr else sp_sigma_y
    elif n == "z":
        sigma = sigma_z if not sympy_expr else sp_sigma_z
    else:
        raise ValueError("n must be 'x' or 'y' or 'z'")

    if sympy_expr:
        return sp.cos(alpha / 2) * identity - 1j * sp.sin(alpha / 2) * sigma
    else:
        return np.cos(alpha / 2) * identity - 1j * np.sin(alpha / 2) * sigma


def ctrl_z(sympy_expr=False):
    """
    Compute controlled-Z gate for 2 qubits.
    :return:
    """
    if sympy_expr:
        cz = sp.eye(4)
        cz[3, 3] = -1
        return cz
    else:
        cz = np.identity(4)
        cz[3, 3] = -1.
        return cz


def calc_multivariate_fourier_series(f: sp.Expr, x: sp.Symbol, y: sp.Symbol, m_max: int, n_max: int) \
        -> tuple[sp.Expr, np.ndarray]:
    """
    Compute the amplitude-phase form of the Fourier series of a real-valued 2*pi periodic function
    of two variables x and y.

    Parameters:
        f: function to compute the Fourier series of
        x: first variable with respect to which to compute the Fourier series
        y: second variable ------------------------"------------------------
        m_max: max. frequency of first variable in the Fourier series
        n_max: max. frequency of second variable ---------"---------

    Returns:
        amplitude-phase form of the Fourier series of the function,
        calculated complex Fourier coefficients
    """
    amp_phase_series = 0
    coeffs = np.zeros((m_max + 1, 2 * n_max + 1), dtype=object)

    for m in range(m_max + 1):
        for n in range(-n_max, n_max + 1):
            # due to symmetry of Fourier coefficients when f is real-valued only positive frequencies m are required
            if m == 0 and n < 0:
                continue

            # compute and save Fourier coefficient
            try:
                c_mn = 1 / (2 * sp.pi) ** 2 \
                       * sp.integrate(sp.integrate(f * sp.exp(- sp.I * (m * x + n * y)),
                                                   (x, -sp.pi, sp.pi)),
                                      (y, -sp.pi, sp.pi))
            except sp.PolynomialDivisionFailed:
                # TODO: clarify whether the following is really sensible
                c_mn = 0.

            if sp.re(c_mn) < 1e-16:
                c_mn = 1j * sp.im(c_mn)
            if sp.im(c_mn) < 1e-16:
                c_mn = sp.re(c_mn)
                # end of TODO

            coeffs[m, n + n_max] = c_mn

            if m == 0 and n > 0:
                coeffs[m, n] = sp.conjugate(c_mn)

            # convert it to amplitude and phase
            if m == 0 and n == 0:
                a_mn = c_mn
                phi_mn = 0.
            else:
                a_mn = 2 * abs(c_mn)
                phi_mn = sp.arg(c_mn)
                phi_mn = 0. if phi_mn == sp.nan else phi_mn

            amp_phase_series += a_mn * sp.cos(m * x + n * y + phi_mn)

    return amp_phase_series, coeffs


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


def calc_mean_squared_difference(data_array_1: np.ndarray, data_array_2: np.ndarray):
    return np.nanmean((data_array_1 - data_array_2) ** 2)


# CLASSES
class FourierCoeffs:
    def __init__(self, no_layers: int):
        """
        # choose all theta angles randomly
        #theta_vector = 2 * np.pi * np.random.random(size=4 * no_layers)
        #print(self.c_omega(theta_vector))

        x, t = sp.symbols("x, t")

        m_max = 1
        n_max = 1

        # thetas = np.pi * np.ones(3)  # np.pi / 4 * np.ones(3)
        # FIXME: calc_expectation_value_1_qubit_n_layers, for example for the 2 problem cases above
        thetas = 2 * np.pi * np.random.random(3)

        no_layers = 1
        expectation_val_z = self.calc_expectation_value_1_qubit_n_layers(no_layers, sp_sigma_z, theta_vals=thetas)

        #print(expectation_val_z)
        #amp_phase_series, coeffs = calc_multivariate_fourier_series(expectation_val_z, x, t, m_max, n_max)
        #print("Amplitude-phase form of Fourier series of f(x, t):")
        #print(amp_phase_series)
        #print(coeffs)

        print(np.round(self.calc_fourier_coeffs_2D_FFT(expectation_val_z, x, t, 1), 8))
        """

        FourierCoeffs.visualize_fourier_coeffs_1_qubit_n_layers(3, sp_sigma_z, 50)

        # TODO: implement formula for Fourier coefficients with variable no. of data uploading layers and
        #  check whether for several combinations of random angles/parameters they are non-zero
        # TODO: think about how to show the result of this numerical experiment in general/analytically or
        #  how to argue for the result


    @staticmethod
    def calc_unitary_transform_1_qubit_1_layer(thetas: np.ndarray, four_thetas: bool, dagger: bool):
        """

        :param thetas:
        :param four_thetas:
        :param sympy_for_thetas:
        :param dagger:
        :return:
        """
        # asserts
        if four_thetas:
            assert len(thetas) == 4, "if four_thetas == True, thetas must be of length 4"
        else:
            assert len(thetas) == 3, "if four_thetas == False, thetas must be of length 3"

        # compute unitary transform for 1 layer
        x, t = sp.symbols("x, t")

        if not dagger:
            unitary_transform = (r_n(thetas[2], "y", sympy_expr=True)
                                 @ r_n(x, "x", sympy_expr=True)
                                 @ r_n(thetas[1], "z", sympy_expr=True)
                                 @ r_n(thetas[0], "y", sympy_expr=True)
                                 @ r_n(t, "x", sympy_expr=True))

            if four_thetas:
                unitary_transform = r_n(thetas[3], "z", sympy_expr=True) @ unitary_transform

        else:
            unitary_transform = (r_n(-t, "x", sympy_expr=True)
                                 @ r_n(-thetas[0], "y", sympy_expr=True)
                                 @ r_n(-thetas[1], "z", sympy_expr=True)
                                 @ r_n(-x, "x", sympy_expr=True)
                                 @ r_n(-thetas[2], "y", sympy_expr=True))

            if four_thetas:
                unitary_transform = unitary_transform @ r_n(-thetas[3], "z", sympy_expr=True)

        return unitary_transform


    @staticmethod
    def calc_expectation_value_1_qubit_n_layers(no_layers: int, obs: np.ndarray, theta_vals: np.ndarray = None):
        """

        :param no_layers:
        :param obs:
        :param theta_vals:
        :return:
        """
        # asserts
        assert no_layers >= 1, "no_layers >= 1 is required"
        assert np.shape(obs) == (2, 2) and np.all(np.conj(np.transpose(obs)) == np.array(obs)),\
            "obs must be a Hermitian 2x2-matrix"

        if theta_vals is not None:
            assert len(theta_vals) == 4 * no_layers - 1, "thetas must be of length 4 * no_layers - 1"

        # compute generic forms of unitary transforms for layers
        alpha, beta, gamma, delta = sp.symbols("alpha, beta, gamma, delta")
        generic_thetas = np.array([alpha, beta, gamma, delta])

        if np.all(np.array(obs) == sigma_z):
            unitary_last_layer = FourierCoeffs.calc_unitary_transform_1_qubit_1_layer(generic_thetas[:3],
                                                                                      False, False)
            unitary_last_layer_dagger = FourierCoeffs.calc_unitary_transform_1_qubit_1_layer(generic_thetas[:3],
                                                                                             False, True)

        if not np.all(np.array(obs) == sigma_z) or no_layers > 1:
            unitary_1_layer = FourierCoeffs.calc_unitary_transform_1_qubit_1_layer(generic_thetas,
                                                                                   True, False)
            unitary_1_layer_dagger = FourierCoeffs.calc_unitary_transform_1_qubit_1_layer(generic_thetas,
                                                                                          True, True)

        # multiply unitary transforms for layers
        if theta_vals is None:
            no_thetas = 4 * no_layers - 1
            thetas = sp.symbols("theta1:" + str(no_thetas + 1))
            symbolic = True
        else:
            thetas = theta_vals
            symbolic = False

        unitary = np.identity(2)
        unitary_dagger = np.identity(2)

        if np.all(np.array(obs) == sigma_z):
            n_max = no_layers - 1
        else:
            n_max = no_layers

        def param_subs(expr: sp.Expr | sp.Matrix, subs_dict: dict, symbolic: bool):
            if symbolic:
                return expr.subs(subs_dict)
            else:
                return expr.evalf(subs=subs_dict, chop=1e-16)

        for n in range(n_max):
            subs_dict = {alpha: thetas[4 * n + 0], beta: thetas[4 * n + 1],
                         gamma: thetas[4 * n + 2], delta: thetas[4 * n + 3]}

            unitary = param_subs(unitary_1_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ param_subs(unitary_1_layer_dagger, subs_dict, symbolic)

        if np.all(np.array(obs) == sigma_z):
            subs_dict = {alpha: thetas[-3], beta: thetas[-2], gamma: thetas[-1]}

            unitary = param_subs(unitary_last_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ param_subs(unitary_last_layer_dagger, subs_dict, symbolic)

        expectation_val = (unitary_dagger @ obs @ unitary)[0, 0]
        # [0, 0] because initial state of quantum circuit is |0>

        return expectation_val
        # return sp.simplify(expectation_val)


    @staticmethod
    def calc_fourier_coeffs_2D_FFT(func: sp.Expr, x: sp.Symbol, y: sp.Symbol, n: int):
        """
        This function consists of code adjusted from the PennyLane demo by Schuld and Meyer
        (https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#part-iii-sampling-fourier-coefficients).
        Computes the first 2*n+1 Fourier coefficients of a 2*pi periodic function.
        """
        lambdified_func = sp.lambdify([x, y], func)

        n_coeffs = 2 * n + 1

        vals = np.linspace(0, 2 * np.pi, n_coeffs, endpoint=False)
        x_mesh, y_mesh = np.meshgrid(vals, vals, indexing="ij")

        z = np.fft.rfftn(lambdified_func(x_mesh, y_mesh)) / vals.size ** 2
        # TODO: clarify reason for factor x.size
        return np.fft.fftshift(z, axes=0)  # such that the zero-frequency coefficient is in the middle


    @staticmethod
    def visualize_fourier_coeffs_1_qubit_n_layers(no_layers: int, obs: np.ndarray, no_random_samples=100):
        """
        This function consists of code adjusted from the PennyLane demo by Schuld and Meyer
        (https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#part-iii-sampling-fourier-coefficients).
        Visualizes the Fourier coefficients of the expectation value of the observable obs
        for the 1-qubit quantum circuit with n data uploading layers.
        """
        # initialize variables
        x, t = sp.symbols("x, t")
        coeffs = []

        # compute Fourier coefficients
        for i in range(no_random_samples):
            thetas = 2 * np.pi * np.random.random(4 * no_layers - 1)
            expectation_val_z = FourierCoeffs.calc_expectation_value_1_qubit_n_layers(no_layers, obs, theta_vals=thetas)

            coeffs_sample = FourierCoeffs.calc_fourier_coeffs_2D_FFT(expectation_val_z, x, t, no_layers)
            coeffs.append(coeffs_sample)

        coeffs = np.array(coeffs)
        coeffs_real = np.real(coeffs)
        coeffs_imag = np.imag(coeffs)

        # plot Fourier coefficients
        no_random_samples, no_x, no_t = np.shape(coeffs)

        fig, ax = plt.subplots(no_x, no_t, figsize=(2 * no_t, 2 * no_x), squeeze=False)

        for m in range(no_x):
            for n in range(no_t):
                ax[m, n].set_title("$c_{" + str(m - no_layers) + str(n) + "}$")
                ax[m, n].scatter(coeffs_real[:, m, n], coeffs_imag[:, m, n], s=20,
                                 facecolor='white', edgecolor='red')
                ax[m, n].set_aspect("equal")
                ax[m, n].set_ylim(-1, 1)
                ax[m, n].set_xlim(-1, 1)

        plt.tight_layout(pad=0.5)
        plt.show()


class ReinforcementLearningFits:
    def __init__(self, reweighted_dynamics: ReweightedDynamics, T: int, no_layers: int, no_fits: int, set_title=True):
        # save inputs
        self.T = T
        self.no_layers = no_layers
        self.no_fits = no_fits

        # initialize arrays
        t_values = np.arange(T)
        x_values = np.arange(- T + 1, T)

        self.coords_array = np.array([[(t, x) for x in x_values]
                                      for t in t_values])

        P_W_array = reweighted_dynamics.P_W_array

        ### plot policy (in the 1-qubit case w/o data RE-uploading)
        ### as SANITY CHECK also in terms of variational parameters \theta_1, \theta_2, and \theta_3)
        # import results of QRL algorithm
        self.policy_array = import_policy_from_csv(T, path='/Users/davidreiss/Desktop/Archiv/Quantum_PG_2/Plots/8_Final_policy_probabilities.csv')
        plot_mask = np.isnan(self.policy_array)
        self.params_1_qubit_array = np.array([1.003468, 1.0284932]
                                             + [0.8490333, 1.8261642, 1.0306203]#, 0.94512033]
                                             + [2.0050492])#, -2.005049])
        # input scalings, variational params, output scaling

        """
        # calculate amplitudes and phases from variational parameters
        params_array = self.calc_params_array_1_qubit_case(params_1_qubit_array)

        # plot policy and prediction from analytical calculation
        self.plot_prob_distribution(T, self.policy_array, "$\pi$ ")
        self.plot_prob_distribution(T, self.softmax_policy(self.coords_array, params_array), "$P_\\theta$ ", masking=True)
        """

        ### fits to reweighted dynamics
        ## fits in terms of variational parameters thetas
        # fits starting from symbolic computation of Fourier coefficients as functions of thetas
        self.fit_and_plot_softmax_policy(self.softmax_policy_thetas, self.coords_array, P_W_array, T, no_fits,
                                         no_thetas=3, set_title=set_title, title="$\pi_\\theta$ Mathematica",
                                         plot_mask=plot_mask, plot_diff=True)

        # fits starting from SymPy expressions

        self.softmax_policy_lambdified = \
            self.softmax_policy_from_sympy_expr(FourierCoeffs.calc_expectation_value_1_qubit_n_layers(self.no_layers,
                                                                                                      sp_sigma_z),
                                                self.no_layers)

        no_fits = 1
        self.fit_and_plot_softmax_policy(self.softmax_policy_from_lambdified_expr, self.coords_array, P_W_array, T,
                                         no_fits, no_thetas=(4 * self.no_layers - 1), set_title=set_title,
                                         title="$\pi_\\theta$ SymPy", plot_mask=plot_mask, plot_diff=True)

        ## fits in terms of Fourier coefficients (amplitudes and phases)
        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        no_fits = 1
        self.fit_and_plot_softmax_policy(self.softmax_policy, self.coords_array, P_W_array, T, no_fits,
                                         no_amplitudes=no_pos_freqs * no_freqs, no_phases=no_pos_freqs * no_freqs,
                                         set_title=set_title, title="$\pi$ amplitudes and phases general", 
                                         plot_mask=plot_mask, plot_diff=True)

        self.fit_and_plot_softmax_policy(self.softmax_policy_1_qubit, self.coords_array, P_W_array, T, no_fits,
                                         no_amplitudes=3, no_phases=3,
                                         set_title=set_title, title="$\pi$ amplitudes and phases 1 qubit",
                                         plot_mask=plot_mask, plot_diff=True)


    @staticmethod
    def calc_params_array_1_qubit_case(params_1_qubit_array: np.ndarray, in_terms_of_thetas=True):
        """

        :param params_1_qubit_array: contains lambdas, either thetas or INDEPENDENT amplitudes and phases, and w
        :param in_terms_of_thetas:
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
            phi_array = np.zeros(2 * 3)

            for i in range(3):
                a_array[1 * 3 + i] = params_1_qubit_array[2 + i]
                phi_array[1 * 3 + i] = params_1_qubit_array[2 + 3 + i]

        # construct params_array suitable for function softmax_policy
        params_array = np.zeros(2 + len(a_array) + len(phi_array) + 1)
        params_array[:2] = params_1_qubit_array[:2]
        params_array[2:-1] = np.concatenate((a_array, phi_array))
        params_array[-1:] = params_1_qubit_array[-1:]

        return params_array


    def softmax_policy(self, coords_array: np.ndarray, params_array: np.ndarray):
        """

        :param coords_array:
        :param params_array:
        :return:
        """
        # split params_array in "subarrays"
        lambda_array = params_array[:2]
        coeffs_array = params_array[2:-1]
        w = params_array[-1:]

        no_pos_freqs = self.no_layers + 1
        no_freqs = 2 * self.no_layers + 1

        # asserts
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
        """

        :param coords_array:
        :param params_1_qubit_array:
        :return:
        """
        return self.softmax_policy(coords_array, self.calc_params_array_1_qubit_case(params_1_qubit_array))


    def softmax_policy_1_qubit(self, coords_array: np.ndarray, params_1_qubit_array: np.ndarray):
        """

        :param coords_array:
        :param params_1_qubit_array:
        :return:
        """
        return self.softmax_policy(coords_array, self.calc_params_array_1_qubit_case(params_1_qubit_array,
                                                                                     in_terms_of_thetas=False))

    @staticmethod
    def softmax_policy_from_sympy_expr(expectation_val: sp.Expr, no_layers: int) \
            -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """

        :param expectation_val:
        :param no_layers:
        :return:
        """
        # define variables
        t, x = sp.symbols("t, x")
        lambda_t, lambda_x = sp.symbols("lambda_t, lambda_x")

        no_thetas = 4 * no_layers - 1
        thetas = list(sp.symbols("theta1:" + str(no_thetas + 1)))

        w = sp.symbols("w")

        # substitute variables
        expectation_val = expectation_val.subs(t, sp.atan(lambda_t * t))
        expectation_val = expectation_val.subs(x, sp.atan(lambda_x * x))

        # apply softmax function
        softmax_policy = 1 / (sp.exp(w * expectation_val) + 1)
        # TODO: check whether this last form is correct!

        # lambdify SymPy expression
        return sp.lambdify([t, x, lambda_t, lambda_x] + thetas + [w], softmax_policy)


    def softmax_policy_from_lambdified_expr(self, coords_array: np.ndarray, params_array: np.ndarray):
        """

        :param coords_array:
        :param params_array:
        :return:
        """
        # coords_array is constructed by np.array([[(t, x) for x in x_values]
        #                                          for t in t_values])
        no_t, no_x, _ = np.shape(coords_array)
        return np.array([[self.softmax_policy_lambdified(coords_array[m, n, 0], coords_array[m, n, 1], *params_array)
                          for n in range(no_x)]
                         for m in range(no_t)])


    @staticmethod
    def fit_and_plot_softmax_policy(softmax_policy: Callable, coords_array: np.ndarray, data_array: np.ndarray,
                                    T: int, no_fits: int, no_thetas: int = None, no_amplitudes: int = None,
                                    no_phases: int = None, set_title=False, title="", plot_mask: np.ndarray = None,
                                    plot_diff=True):
        """

        :param softmax_policy:
        :param coords_array:
        :param data_array:
        :param T:
        :param no_fits:
        :param no_thetas:
        :param no_amplitudes:
        :param no_phases:
        :param set_title:
        :param title:
        :param plot_mask:
        :param plot_diff:
        :return:
        """
        assert no_fits > 0, "no_fits must be an integer greater than 0, otherwise the function does not make sense"

        residual_mean_squared_error_list = []
        optimized_params_list = []

        # initialize instance progress_bar of utility class ProgressBar
        progress_bar = ProgressBar(no_fits, "Fits of policy starting from different random choices of parameters")

        for i in range(no_fits):
            # update progress_bar due to progress
            progress_bar.update(i)

            if no_thetas is not None:
                initial_scalings = np.random.standard_normal(3)
                initial_thetas = 2 * np.pi * np.random.random(no_thetas)
                initial_params = np.insert(initial_scalings, 2, initial_thetas)

                bounds_params = ([(-np.inf, np.inf)] * 2
                                 + [(0., 2 * np.pi)] * no_thetas
                                 + [(-np.inf, np.inf)])

            if no_amplitudes is not None and no_phases is not None:
                initial_scalings = np.random.standard_normal(3 + no_amplitudes)
                initial_phases = 2 * np.pi * np.random.random(no_phases)
                initial_params = np.insert(initial_scalings, 2 + no_amplitudes, initial_phases)

                bounds_params = ([(-np.inf, np.inf)] * (2 + no_amplitudes)
                                 + [(0., 2 * np.pi)] * no_phases
                                 + [(-np.inf, np.inf)])

            optimized_params, residual_mean_squared_error = \
                fit_multivariate_func_leastsq(softmax_policy, coords_array, data_array,
                                              params_initial_guess=initial_params,
                                              params_bounds=bounds_params,
                                              no_independent_vars=2)

            residual_mean_squared_error_list.append(residual_mean_squared_error)
            optimized_params_list.append(optimized_params)

        # finish progress bar
        progress_bar.finish()

        index = np.argmin(residual_mean_squared_error_list)
        residual_mean_squared_error_min = residual_mean_squared_error_list[index]
        optimized_params_min = optimized_params_list[index]

        vals_fitted_func_array = softmax_policy(coords_array, optimized_params_min)

        if plot_diff:
            plot_prob_distribution(T, np.abs(vals_fitted_func_array - data_array),
                                   set_title=set_title, title=title,
                                   plot_mask=plot_mask, diff=plot_diff)
        else:
            plot_prob_distribution(T, vals_fitted_func_array,
                                   set_title=set_title, title=title,
                                   plot_mask=plot_mask, diff=plot_diff)

        print("smallest residual_mean_squared_error: ", residual_mean_squared_error_min)

        if no_fits > 1:
            print("second smallest residual_mean_squared_error: ",
                  min(residual_mean_squared_error_list.remove(residual_mean_squared_error_min)))

        return
