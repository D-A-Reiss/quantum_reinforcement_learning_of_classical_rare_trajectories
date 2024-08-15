# IMPORTS #####
import os
from multiprocessing import Pool

import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import orqviz

from typing import Callable, List
from scipy.optimize import minimize
from sympy.physics.quantum import TensorProduct

from reweighted_dynamics import ReweightedDynamics
from utilities import import_policy_from_csv, einsum_subscripts, ProgressBar, plot_prob_distribution, \
    write_plot_params_to_file


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
plt.rcParams['axes.unicode_minus'] = False

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
def tensor_prod(a, b, sympy_expr=False):
    if sympy_expr:
        return TensorProduct(a, b)
    else:
        return np.kron(a, b)


def r_n(alpha: float | sp.Symbol, n: str, sympy_expr=False, deactivated=False) -> np.ndarray:
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

    if deactivated:
        return identity

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


def r_n_multi_qubit(tot_no_qubits: int, acting_on_qubit_no: int, alpha: float | sp.Symbol, n: str, sympy_expr=False,
                    deactivated=False) -> np.ndarray:
    assert tot_no_qubits > 1, "tot_no_qubits must be greater than 1"

    identity = np.identity(2) if not sympy_expr else sp_identity
    single_qubit_r_n = r_n(alpha, n, sympy_expr=sympy_expr, deactivated=deactivated)

    def choose_nth_matrix(n):
        return single_qubit_r_n if acting_on_qubit_no == n else identity

    matrix = tensor_prod(choose_nth_matrix(1), choose_nth_matrix(2), sympy_expr=sympy_expr)

    for n in range(3, tot_no_qubits + 1):
        matrix = tensor_prod(matrix, choose_nth_matrix(n), sympy_expr=sympy_expr)

    return matrix


def ctrl_z(sympy_expr=False, deactivated=False):
    """
    Compute controlled-Z gate for 2 qubits.
    :return:
    """
    if sympy_expr:
        cz = sp.eye(4)
    else:
        cz = np.identity(4)

    if deactivated:
        return cz

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


def fit_func_to_data(func: Callable[[np.ndarray, np.ndarray], np.ndarray], coords_array: np.ndarray,
                     data_array: np.ndarray, params_initial_guess: np.ndarray, params_bounds=None,
                     asserts=True, no_independent_vars: int = None, cost_func_type="leastsq",
                     no_trajectories: int = None, T: int = None, s: float = None):
    """
    Fit of multivariate function func at coordinates in coords_array to data_array
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

        if cost_func_type == "trajectory_KL_divergence":
            assert no_trajectories is not None, \
                'if cost_func_type == "trajectory_KL_divergence", no_trajectories must be provided'
            assert T is not None, 'if cost_func_type == "trajectory_KL_divergence", T must be provided'
            assert s is not None, 'if cost_func_type == "trajectory_KL_divergence", s must be provided'

    # define cost function
    if cost_func_type == "leastsq":
        def cost_func(params):
            # compute values of func for coords_array and params
            func_vals_array = func(coords_array, params)

            # compute and return mean squared error as cost, ignoring NaNs
            return np.nanmean((func_vals_array - data_array) ** 2)

    elif cost_func_type == "trajectory_KL_divergence":
        # generate trajectories for P_W
        trajectories_x_array_P_W = PolicyEvaluation.calc_trajectories_x_array(no_trajectories, T, data_array)

        # compute return values and estimate average return for P_W
        return_values_list = PolicyEvaluation.calc_return_values(trajectories_x_array_P_W, T, data_array, s)
        estimate_avg_return_P_W = sum(return_values_list) / len(return_values_list)

        def cost_func(params):
            # compute values of func for coords_array and params
            func_vals_array = func(coords_array, params)

            # generate trajectories for P_theta
            trajectories_x_array_P_theta = PolicyEvaluation.calc_trajectories_x_array(no_trajectories, T,
                                                                                      func_vals_array)

            # compute return values and estimate average return for P_theta
            return_values_list = PolicyEvaluation.calc_return_values(trajectories_x_array_P_theta, T, func_vals_array,
                                                                     s)
            estimate_avg_return_P_theta = sum(return_values_list) / len(return_values_list)

            # compute Kullback-Leibler divergence
            estimate_KL_divergence = estimate_avg_return_P_W - estimate_avg_return_P_theta

            if estimate_KL_divergence < 0:
                raise ValueError("KL divergence estimate is negative, so no_trajectories is chosen too small")
            else:
                return estimate_KL_divergence

    else:
        raise NotImplementedError(f'Option cost_func_type="{cost_func_type}" has not been implemented yet.')

    result = minimize(cost_func, params_initial_guess, bounds=params_bounds)

    print("\n")
    print("Fitting was successful:", result.success)

    optimized_params = result.x
    residual_mean_squared_error = result.fun

    return optimized_params, residual_mean_squared_error


def calc_mean_squared_difference(data_array_1: np.ndarray, data_array_2: np.ndarray):
    return np.nanmean((data_array_1 - data_array_2) ** 2)


# CLASSES
class FourierCoeffs:
    def __init__(self, no_qubits: int, no_layers: int, no_samples: int, random_thetas=True,
                 optimized_fourier_coeffs: np.ndarray = None, deactivate_r_x=False, deactive_r_y_and_r_z=False,
                 deactive_ctrl_z=False):
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

        if no_qubits == 1:
            obs = sp_sigma_z
        elif no_qubits == 2:
            obs = tensor_prod(sp_sigma_z, sp_sigma_z, sympy_expr=True)
        else:
            pass
            # implement this if needed

        """
        expectation_val_z_0_z_1 = self.calc_expectation_value_2_qubits_n_layers(no_layers, obs)
        print(expectation_val_z_0_z_1)
        print(sp.simplify(expectation_val_z_0_z_1))
        """

        FourierCoeffs.visualize_fourier_coeffs_m_qubits_n_layers(no_qubits, no_layers, obs, no_samples,
                                                                 random_thetas=random_thetas,
                                                                 optimized_fourier_coeffs=optimized_fourier_coeffs)


    @staticmethod
    def param_subs(expr: sp.Expr | sp.Matrix, subs_dict: dict, symbolic: bool):
        if symbolic:
            return expr.subs(subs_dict)
        else:
            return expr.evalf(subs=subs_dict, chop=1e-16)


    @staticmethod
    def calc_unitary_transform_1_qubit_1_layer(thetas: np.ndarray, four_thetas: bool, dagger: bool):
        """

        :param thetas:
        :param four_thetas:
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


        for n in range(n_max):
            subs_dict = {alpha: thetas[4 * n + 0], beta: thetas[4 * n + 1],
                         gamma: thetas[4 * n + 2], delta: thetas[4 * n + 3]}

            unitary = FourierCoeffs.param_subs(unitary_1_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierCoeffs.param_subs(unitary_1_layer_dagger, subs_dict, symbolic)

        if np.all(np.array(obs) == sigma_z):
            subs_dict = {alpha: thetas[-3], beta: thetas[-2], gamma: thetas[-1]}

            unitary = FourierCoeffs.param_subs(unitary_last_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierCoeffs.param_subs(unitary_last_layer_dagger, subs_dict, symbolic)

        expectation_val = (unitary_dagger @ obs @ unitary)[0, 0]
        # [0, 0] because initial state of quantum circuit is |0>

        return expectation_val
        # return sp.simplify(expectation_val)


    @staticmethod
    def calc_unitary_transform_2_qubits_1_layer(thetas: np.ndarray, four_thetas: bool, dagger: bool):
        """

        :param thetas:
        :param dagger:
        :return:
        """
        # asserts
        if four_thetas:
            assert len(thetas) == 4, "if four_thetas == True, thetas must be of length 4"
        else:
            assert len(thetas) == 2, "if four_thetas == False, thetas must be of length 3"

        # compute unitary transform for 1 layer
        x, t = sp.symbols("x, t")

        if not dagger:
            unitary_transform = (r_n_multi_qubit(2, 2, thetas[1], "y", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 1, thetas[0], "y", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 2, x, "x", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 1, t, "x", sympy_expr=True))

            if four_thetas:
                unitary_transform = (ctrl_z(sympy_expr=True)
                                     @ r_n_multi_qubit(2, 2, thetas[3], "z", sympy_expr=True)
                                     @ r_n_multi_qubit(2, 1, thetas[2], "z", sympy_expr=True)
                                     @ unitary_transform)

        else:
            unitary_transform = (r_n_multi_qubit(2, 1, -t, "x", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 2, -x, "x", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 1, -thetas[0], "y", sympy_expr=True)
                                 @ r_n_multi_qubit(2, 2, -thetas[1], "y", sympy_expr=True))

            if four_thetas:
                unitary_transform = (unitary_transform @
                                     r_n_multi_qubit(2, 1, -thetas[2], "z", sympy_expr=True)
                                     @ r_n_multi_qubit(2, 2, -thetas[3], "z", sympy_expr=True)
                                     @ ctrl_z(sympy_expr=True))

        return unitary_transform


    @staticmethod
    def calc_expectation_value_2_qubits_n_layers(no_layers: int, obs: np.ndarray, theta_vals: np.ndarray = None):
        """

        :param no_layers:
        :param obs:
        :param theta_vals:
        :return:
        """
        # asserts
        assert no_layers >= 1, "no_layers >= 1 is required"
        assert np.shape(obs) == (4, 4) and np.all(np.conj(np.transpose(obs)) == np.array(obs)), \
            "obs must be a Hermitian 4x4-matrix"

        if theta_vals is not None:
            assert len(theta_vals) == 4 * no_layers - 2, "thetas must be of length 4 * no_layers - 2"

        # compute generic forms of unitary transforms for layers
        alpha, beta, gamma, delta = sp.symbols("alpha, beta, gamma, delta")
        generic_thetas = np.array([alpha, beta, gamma, delta])

        if np.all(np.array(obs) == tensor_prod(sigma_z, sigma_z)):
            unitary_last_layer = FourierCoeffs.calc_unitary_transform_2_qubits_1_layer(generic_thetas[:2],
                                                                                       False, False)
            unitary_last_layer_dagger = FourierCoeffs.calc_unitary_transform_2_qubits_1_layer(generic_thetas[:2],
                                                                                              False, True)

        if not np.all(np.array(obs) == tensor_prod(sigma_z, sigma_z)) or no_layers > 1:
            unitary_1_layer = FourierCoeffs.calc_unitary_transform_2_qubits_1_layer(generic_thetas,
                                                                                    True, False)
            unitary_1_layer_dagger = FourierCoeffs.calc_unitary_transform_2_qubits_1_layer(generic_thetas,
                                                                                           True, True)

        # multiply unitary transforms for layers
        if theta_vals is None:
            no_thetas = 4 * no_layers - 2
            thetas = sp.symbols("theta1:" + str(no_thetas + 1))
            symbolic = True
        else:
            thetas = theta_vals
            symbolic = False

        unitary = np.identity(4)
        unitary_dagger = np.identity(4)

        if np.all(np.array(obs) == tensor_prod(sigma_z, sigma_z)):
            n_max = no_layers - 1
        else:
            n_max = no_layers


        for n in range(n_max):
            subs_dict = {alpha: thetas[4 * n + 0], beta: thetas[4 * n + 1],
                         gamma: thetas[4 * n + 2], delta: thetas[4 * n + 3]}

            unitary = FourierCoeffs.param_subs(unitary_1_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierCoeffs.param_subs(unitary_1_layer_dagger, subs_dict, symbolic)

        if np.all(np.array(obs) == tensor_prod(sigma_z, sigma_z)):
            subs_dict = {alpha: thetas[-3], beta: thetas[-2], gamma: thetas[-1]}

            unitary = FourierCoeffs.param_subs(unitary_last_layer, subs_dict, symbolic) @ unitary
            unitary_dagger = unitary_dagger @ FourierCoeffs.param_subs(unitary_last_layer_dagger, subs_dict, symbolic)

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
    def visualize_fourier_coeffs_m_qubits_n_layers(no_qubits: int, no_layers: int, obs: np.ndarray, no_samples=100,
                                                   random_thetas=True, optimized_fourier_coeffs: np.ndarray = None):
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
        progress_bar = ProgressBar(no_samples, "Fourier coefficients of 1-qubit-PQC with " + str(no_layers) +
                                   " layers for different choices of thetas")

        if random_thetas:
            for i in range(no_samples):
                progress_bar.update(i)

                if no_qubits == 1:
                    thetas = 2 * np.pi * np.random.random(4 * no_layers - 1)
                    expectation_val_z = FourierCoeffs.calc_expectation_value_1_qubit_n_layers(no_layers, obs,
                                                                                              theta_vals=thetas)

                elif no_qubits == 2:
                    thetas = 2 * np.pi * np.random.random(4 * no_layers)
                    expectation_val_z = FourierCoeffs.calc_expectation_value_2_qubits_n_layers(no_layers, obs,
                                                                                               theta_vals=thetas)

                coeffs_sample = FourierCoeffs.calc_fourier_coeffs_2D_FFT(expectation_val_z, x, t, no_layers)
                coeffs.append(coeffs_sample)

        else:
            if no_qubits == 1:
                no_thetas = 4 * no_layers - 1

            elif no_qubits == 2:
                no_thetas = 4 * no_layers

            theta_vals_array = np.linspace(0., 2 * np.pi, num=int(no_samples**(1 / no_thetas)), endpoint=False)

            theta_vals_array_list = [theta_vals_array] * no_thetas
            theta_meshgrids_list = np.meshgrid(*theta_vals_array_list, indexing="ij")

            theta_vecs_array = np.vstack([meshgrid.ravel() for meshgrid in theta_meshgrids_list]).T

            i = 0

            for theta_vec in theta_vecs_array:
                progress_bar.update(i)

                if no_qubits == 1:
                    expectation_val_z = FourierCoeffs.calc_expectation_value_1_qubit_n_layers(no_layers, obs,
                                                                                              theta_vals=theta_vec)

                elif no_qubits == 2:
                    expectation_val_z = FourierCoeffs.calc_expectation_value_2_qubits_n_layers(no_layers, obs,
                                                                                               theta_vals=theta_vec)

                coeffs_sample = FourierCoeffs.calc_fourier_coeffs_2D_FFT(expectation_val_z, x, t, no_layers)
                coeffs.append(coeffs_sample)

                i += 1

        progress_bar.finish()

        coeffs = np.array(coeffs)
        coeffs_real = np.real(coeffs)
        coeffs_imag = np.imag(coeffs)

        no_samples, no_x, no_t = np.shape(coeffs)

        if optimized_fourier_coeffs is not None:
            no_opt_coeffs = len(optimized_fourier_coeffs)
            opt_coeffs_real = 1 / 2 * optimized_fourier_coeffs[:no_opt_coeffs // 2] \
                              * np.cos(optimized_fourier_coeffs[no_opt_coeffs // 2:])
            opt_coeffs_imag = 1 / 2 * optimized_fourier_coeffs[:no_opt_coeffs // 2] \
                              * np.sin(optimized_fourier_coeffs[no_opt_coeffs // 2:])

            opt_coeffs_real = np.swapaxes(opt_coeffs_real.reshape(no_t, no_x), 0, 1)
            opt_coeffs_imag = np.swapaxes(opt_coeffs_imag.reshape(no_t, no_x), 0, 1)

            # multiply c_{00} by 2
            opt_coeffs_real[(no_x - 1) // 2, 0] *= 2
            opt_coeffs_imag[(no_x - 1) // 2, 0] *= 2

            opt_coeffs_real = np.expand_dims(opt_coeffs_real, 0)
            opt_coeffs_imag = np.expand_dims(opt_coeffs_imag, 0)

        # plot Fourier coefficients
        if no_layers == 1:
            fig, ax = plt.subplots(no_x, no_t, figsize=(2 * no_t, 2 * no_x), squeeze=False)

            for m in range(no_x):
                for n in range(no_t):
                    ax[m, n].set_title("$c_{" + str(m - no_layers) + str(n) + "}$")
                    ax[m, n].scatter(coeffs_real[:, m, n], coeffs_imag[:, m, n], s=20,
                                     facecolor='white', edgecolor='red')
                    ax[m, n].scatter(opt_coeffs_real[:, m, n], opt_coeffs_imag[:, m, n], s=20,
                                     facecolor='white', edgecolor='black')
                    ax[m, n].set_aspect("equal")
                    ax[m, n].set_ylim(-1, 1)
                    ax[m, n].set_xlim(-1, 1)

        else:
            fig, ax = plt.subplots(no_t, no_x, figsize=(2 * no_x, 2 * no_t), squeeze=False)

            for m in range(no_x):
                for n in range(no_t):
                    ax[n, m].set_title("$c_{" + str(m - no_layers) + str(n) + "}$")
                    ax[n, m].scatter(coeffs_real[:, m, n], coeffs_imag[:, m, n], s=20,
                                     facecolor='white', edgecolor='green')
                    ax[n, m].scatter(opt_coeffs_real[:, m, n], opt_coeffs_imag[:, m, n], s=20,
                                     facecolor='white', edgecolor='black')
                    ax[n, m].set_aspect("equal")
                    ax[n, m].set_ylim(-1, 1)
                    ax[n, m].set_xlim(-1, 1)
                    #ax[n, m].set_ylim(-2, 2)
                    #ax[n, m].set_xlim(-2, 2)

        plt.tight_layout(pad=0.5)

        # save plot
        fig.savefig("Fourier_coeffs_" + str(no_qubits) + "_qubits_" + str(no_layers) + "_layers.pdf",
                    bbox_inches="tight")

        plt.show()


class ReinforcementLearningFits:
    def __init__(self, reweighted_dynamics: ReweightedDynamics, T: int, no_layers: int, no_fits: int,
                 no_trajectories: int, no_qubits=1, set_title=True, theta_fits=True,
                 optimized_params_fourier_coeffs: np.ndarray=None, optimized_no_layers: int = None,
                 compute_in_parallel=False, cost_func_type="leastsq"):
        """
        # TODO
        :param reweighted_dynamics:
        :param T:
        :param no_layers:
        :param no_fits:
        :param no_trajectories:
        :param set_title:
        :param theta_fits:
        :param optimized_params_fourier_coeffs:
        :param optimized_no_layers:
        """
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
        plot_mask = np.isnan(P_W_array)
        s = reweighted_dynamics.s

        ### plot policy (in the 1-qubit case w/o data RE-uploading)
        ### as SANITY CHECK also in terms of variational parameters \theta_1, \theta_2, and \theta_3)
        # import results of QRL algorithm
        """
        self.policy_array = import_policy_from_csv(T, path='/Users/davidreiss/Desktop/Archiv/Quantum_PG_2/Plots/8_Final_policy_probabilities.csv')
        plot_mask = np.isnan(self.policy_array)
        self.params_1_qubit_array = np.array([1.003468, 1.0284932]
                                             + [0.8490333, 1.8261642, 1.0306203]#, 0.94512033]
                                             + [2.0050492])#, -2.005049])
        # input scalings, variational params, output scaling
        """
        """
        # calculate amplitudes and phases from variational parameters
        params_array = self.calc_params_array_1_qubit_case(params_1_qubit_array)

        # plot policy and prediction from analytical calculation
        self.plot_prob_distribution(T, self.policy_array, "$\pi$ ")
        self.plot_prob_distribution(T, self.softmax_policy(self.coords_array, params_array), "$P_\\theta$ ", masking=True)
        """

        ### fits to reweighted dynamics
        file_name = str(no_layers) + "_layers_" + str(no_fits) + "_fits"

        ## fits in terms of variational parameters thetas
        if theta_fits:
            # fits starting from symbolic computation of Fourier coefficients as functions of thetas
            if no_layers == 1:
                self.vals_fitted_func_1_qubit_1_layer_thetas, self.optimized_params_1_qubit_1_layer_thetas, \
                    self.residual_mean_squared_errors_1_qubit_1_layer_thetas = \
                    self.fit_and_plot_softmax_policy("Mathematica_1_qubit_" + file_name,
                                                     self.softmax_policy_1_qubit_1_layer_thetas, self.coords_array,
                                                     P_W_array, T, s, 1, no_layers, no_fits, no_trajectories,
                                                     no_thetas=3, set_title=set_title, plot_mask=plot_mask,
                                                     plot_diff=True, compute_in_parallel=compute_in_parallel,
                                                     cost_func_type=cost_func_type)

            # fits starting from SymPy expressions
            if no_qubits == 1:
                self.softmax_policy_lambdified = \
                    self.softmax_policy_from_sympy_expr(
                        FourierCoeffs.calc_expectation_value_1_qubit_n_layers(self.no_layers, sp_sigma_z),
                        self.no_layers)

            elif no_qubits == 2:
                self.softmax_policy_lambdified = \
                    self.softmax_policy_from_sympy_expr(
                        FourierCoeffs.calc_expectation_value_2_qubits_n_layers(self.no_layers,
                                                                               tensor_prod(sp_sigma_z, sp_sigma_z,
                                                                                           sympy_expr=True)),
                        self.no_layers)

            else:
                raise NotImplementedError("Case no_qubits > 2 has not been implemented yet.")

            self.vals_fitted_func_from_sympy, self.optimized_params_from_sympy, \
                self.residual_mean_squared_errors_from_sympy = \
                self.fit_and_plot_softmax_policy("SymPy_" + str(no_qubits) + "_qubits_" + file_name,
                                                 self.softmax_policy_from_lambdified_expr, self.coords_array, P_W_array,
                                                 T, s, 1, no_layers, no_fits, no_trajectories,
                                                 no_thetas=(4 * self.no_layers - 1), set_title=set_title,
                                                 plot_mask=plot_mask, plot_diff=True,
                                                 compute_in_parallel=compute_in_parallel, cost_func_type=cost_func_type)

        ## fits in terms of Fourier coefficients (amplitudes and phases)
        else:
            no_pos_freqs = no_layers + 1
            no_freqs = 2 * no_layers + 1

            if no_layers == 1:
                self.vals_fitted_func_1_qubit_fourier_coeffs, self.optimized_params_1_qubit_1_layer_fourier_coeffs, \
                    self.residual_mean_squared_errors_1_qubit_1_layer_fourier_coeffs = \
                    self.fit_and_plot_softmax_policy("Fourier_coeffs_restricted_1_qubit_" + file_name,
                                                     self.softmax_policy_1_qubit_1_layer_fourier_coeffs, self.coords_array,
                                                     P_W_array, T, s, 1, no_layers, no_fits, no_trajectories,
                                                     no_amplitudes=3, no_phases=3, set_title=set_title,
                                                     plot_mask=plot_mask, plot_diff=True,
                                                     compute_in_parallel=compute_in_parallel,
                                                     cost_func_type=cost_func_type)

                self.vals_fitted_func_2_qubits_fourier_coeffs, \
                    self.optimized_params_2_qubits_1_layer_fourier_coeffs, \
                    self.residual_mean_squared_errors_2_qubits_1_layer_fourier_coeffs = \
                    self.fit_and_plot_softmax_policy("Fourier_coeffs_restricted_2_qubits_" + file_name,
                                                     self.softmax_policy_2_qubits_1_layer_fourier_coeffs, self.coords_array,
                                                     P_W_array, T, s, 2, no_layers, no_fits, no_trajectories,
                                                     no_amplitudes=1, no_phases=0, set_title=set_title,
                                                     plot_mask=plot_mask, plot_diff=True,
                                                     compute_in_parallel=compute_in_parallel,
                                                     cost_func_type=cost_func_type)

            else:
                self.vals_fitted_func_1_qubit_fourier_coeffs, self.optimized_params_1_qubit_fourier_coeffs, \
                    self.residual_mean_squared_errors_1_qubit_fourier_coeffs = \
                    self.fit_and_plot_softmax_policy("1_and_2_qubits_" + file_name, self.softmax_policy, self.coords_array,
                                                     P_W_array, T, s, 1, no_layers, no_fits, no_trajectories,
                                                     no_amplitudes=no_pos_freqs * no_freqs,
                                                     no_phases=no_pos_freqs * no_freqs,
                                                     set_title=set_title, plot_mask=plot_mask, plot_diff=True,
                                                     optimized_params_fourier_coeffs=optimized_params_fourier_coeffs,
                                                     optimized_no_layers=optimized_no_layers,
                                                     compute_in_parallel=compute_in_parallel,
                                                     cost_func_type=cost_func_type)


    @staticmethod
    def calc_params_array_1_qubit_1_layer(params_1_qubit_array: np.ndarray, in_terms_of_thetas=True):
        """

        :param params_1_qubit_array: contains lambdas, either thetas or INDEPENDENT amplitudes and phases, and w
        :param in_terms_of_thetas:
        :return:
        """
        # asserts
        if in_terms_of_thetas:
            thetas_array = params_1_qubit_array[2:-1]
            assert len(thetas_array) == 3, \
                "amplitudes and phases in truncated Fourier series are determined in the 1-qubit-1-layer case " \
                "by 3 angles theta_1, theta_2, and theta_3, which must be supplied in params_1_qubit_array[2:-1]"
        else:
            assert len(params_1_qubit_array[2:-1]) == 6, \
                "amplitudes and phases in truncated Fourier series are determined in the 1-qubit-1-layer case " \
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


    @staticmethod
    def calc_params_array_2_qubits_1_layer(params_2_qubits_array: np.ndarray, in_terms_of_thetas=True):
        """

        :param params_2_qubits_array: contains lambdas, either thetas or INDEPENDENT amplitudes and phases, and w
        :param in_terms_of_thetas:
        :return:
        """
        # asserts
        if in_terms_of_thetas:
            raise NotImplementedError("implementation of case in_terms_of_thetas == True hasn't been finished yet")
            thetas_array = params_2_qubits_array[2:-1]
            assert len(thetas_array) == 4, \
                "amplitudes and phases in truncated Fourier series are determined in the 2-qubits-1-layer case " \
                "by 4 angles theta_1, theta_2, theta_3, and theta_4 which must be supplied in " \
                "params_2_qubits_array[2:-1]"
        else:
            assert len(params_2_qubits_array[2:-1]) == 1, \
                "amplitudes and phases in truncated Fourier series are determined in the 2-qubits-1-layer case " \
                "by 1 non-zero amplitudes, which must be supplied in this order in params_2_qubits_array[2:-1]"

        if in_terms_of_thetas:
            theta_1 = thetas_array[0]
            theta_2 = thetas_array[1]
            theta_3 = thetas_array[2]
            theta_4 = thetas_array[3]

            # complex-valued coefficients of truncated Fourier series
            c_array = np.zeros(2 * 3, dtype=complex)
            #c_array[1 * 3 + 0] =
            #c_array[1 * 3 + 2] =
            # TODO: to be computed and implemented

            # corresponding amplitudes and phases in truncated Fourier series
            a_array = 2 * np.abs(c_array)
            phi_array = np.angle(c_array)
        else:
            a_array = np.zeros(2 * 3)
            phi_array = np.zeros(2 * 3)

            a_array[1 * 3 + 0] = params_2_qubits_array[2]
            a_array[1 * 3 + 2] = params_2_qubits_array[2]

        # construct params_array suitable for function softmax_policy
        params_array = np.zeros(2 + len(a_array) + len(phi_array) + 1)
        params_array[:2] = params_2_qubits_array[:2]
        params_array[2:-1] = np.concatenate((a_array, phi_array))
        params_array[-1:] = params_2_qubits_array[-1:]

        return params_array


    @staticmethod
    def get_subarrays_from_params_array(params_array: np.ndarray, no_layers: int):
        lambda_array = params_array[:2]
        coeffs_array = params_array[2:-1]
        w = params_array[-1:]

        no_pos_freqs = no_layers + 1
        no_freqs = 2 * no_layers + 1

        amplitudes_array = coeffs_array[:no_pos_freqs * no_freqs].reshape(no_pos_freqs, no_freqs)
        phases_array = coeffs_array[no_pos_freqs * no_freqs:].reshape(no_pos_freqs, no_freqs)

        return lambda_array, w, amplitudes_array, phases_array


    @staticmethod
    def get_params_array_from_subarrays(lambda_array: np.ndarray, w: np.ndarray, amplitudes_array: np.ndarray,
                                        phases_array: np.ndarray):
        return np.concatenate((lambda_array, amplitudes_array.flatten(), phases_array.flatten(), w))


    @staticmethod
    def insert_optimized_params_into_larger_params_array(optimized_params_array: np.ndarray, params_array: np.ndarray,
                                                         optimized_no_layers: int, no_layers: int):
        opt_params_array = optimized_params_array
        opt_no_layers = optimized_no_layers

        opt_lambda_array, opt_w, opt_amplitudes_array, opt_phases_array = \
            ReinforcementLearningFits.get_subarrays_from_params_array(opt_params_array, opt_no_layers)
        lambda_array, w, amplitudes_array, phases_array = \
            ReinforcementLearningFits.get_subarrays_from_params_array(params_array, no_layers)

        lambda_array = opt_lambda_array
        w = opt_w

        amplitudes_array[:(opt_no_layers + 1), (no_layers - opt_no_layers):(no_layers + opt_no_layers + 1)] = \
            opt_amplitudes_array

        phases_array[:(opt_no_layers + 1), (no_layers - opt_no_layers):(no_layers + opt_no_layers + 1)] = \
            opt_phases_array

        return ReinforcementLearningFits.get_params_array_from_subarrays(lambda_array, w, amplitudes_array, phases_array)


    def softmax_policy(self, coords_array: np.ndarray, params_array: np.ndarray):
        """

        :param coords_array:
        :param params_array:
        :return:
        """
        no_pos_freqs = self.no_layers + 1
        no_freqs = 2 * self.no_layers + 1

        # asserts
        assert len(params_array) == 2 + 2 * no_pos_freqs * no_freqs + 1, \
            f"length {len(params_array)} of params_array not correct; it must contain 2 input scaling parameters, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) amplitudes, " \
            "(self.no_layers + 1) * (2 * self.no_layers + 1) phases, " \
            f"and 1 output scaling parameter, here in total: {2 + 2 * no_pos_freqs * no_freqs + 1}"
        # TODO: implement further asserts

        # initializations
        lambda_array, w, amplitudes_array, phases_array = self.get_subarrays_from_params_array(params_array,
                                                                                               self.no_layers)

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

        amplitudes_array = np.broadcast_to(amplitudes_array, (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))
        phases_array = np.broadcast_to(phases_array, (*np.shape(coords_array)[:-1], no_pos_freqs, no_freqs))

        avg_Z = amplitudes_array * np.cos(np.einsum(einsum_subscripts("ft,fx",
                                                                      "gt,gx",
                                                                      to="gt,gx,ft,fx"),
                                                    freqs_t,
                                                    g_t)
                                          + np.einsum(einsum_subscripts("ft,fx",
                                                                        "gt,gx",
                                                                        to="gt,gx,ft,fx"),
                                                      freqs_x,
                                                      g_x)
                                          + phases_array)

        return 1 / (np.exp(w * np.einsum(einsum_subscripts("gt,gx,ft,fx",
                                                           to="gt,gx"),
                                         avg_Z))
                    + 1)
        # TODO: check whether this last form is correct!


    def softmax_policy_1_qubit_1_layer_thetas(self, coords_array: np.ndarray, params_1_qubit_array: np.ndarray):
        """

        :param coords_array:
        :param params_1_qubit_array:
        :return:
        """
        return self.softmax_policy(coords_array, self.calc_params_array_1_qubit_1_layer(params_1_qubit_array,
                                                                                        in_terms_of_thetas=True))


    def softmax_policy_1_qubit_1_layer_fourier_coeffs(self, coords_array: np.ndarray, params_1_qubit_array: np.ndarray):
        """

        :param coords_array:
        :param params_1_qubit_array:
        :return:
        """
        return self.softmax_policy(coords_array, self.calc_params_array_1_qubit_1_layer(params_1_qubit_array,
                                                                                        in_terms_of_thetas=False))

    def softmax_policy_2_qubits_1_layer_fourier_coeffs(self, coords_array: np.ndarray,
                                                       params_2_qubits_array: np.ndarray):
        """

        :param coords_array:
        :param params_2_qubits_array:
        :return:
        """
        return self.softmax_policy(coords_array, self.calc_params_array_2_qubits_1_layer(params_2_qubits_array,
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
    def fit_and_plot_softmax_policy(file_name: str, softmax_policy: Callable, coords_array: np.ndarray,
                                    data_array: np.ndarray, T: int, s: float, no_qubits: int, no_layers: int,
                                    no_fits: int, no_trajectories: int, no_thetas: int = None,
                                    no_amplitudes: int = None, no_phases: int = None, set_title=False,
                                    plot_mask: np.ndarray = None, plot_diff=True,
                                    optimized_params_fourier_coeffs: np.ndarray = None, optimized_no_layers: int = None,
                                    compute_in_parallel=False, cost_func_type="leastsq"):
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

        mean_squared_error_list = []
        optimized_params_list = []

        # initialize instance progress_bar of utility class ProgressBar
        progress_bar = ProgressBar(no_fits, "Fits for " + file_name +
                                   " starting from different random choices of parameters")

        single_job_params = [no_thetas, no_amplitudes, no_phases, optimized_params_fourier_coeffs,
                             optimized_no_layers, no_layers, softmax_policy, coords_array, data_array, cost_func_type,
                             no_trajectories, T, s]
        # it remains to append #fits (to be done by single job) to single_job_params

        if compute_in_parallel:
            cpu_count = os.cpu_count()
            job_size = no_fits // (cpu_count - 1)
            rest_size = no_fits % (cpu_count - 1)
            job_params = []

            for cpu_index in range(cpu_count):
                # compute #fits to be done by single job
                single_job_no_fits = job_size if cpu_index < (cpu_count - 1) else rest_size

                # append #fits (to be done by single job) to single_job_params
                # and append single_job_params to job_params
                job_params.append(single_job_params + [single_job_no_fits])

            progress_bar.update()

            with Pool() as pool:
                for task_result in pool.imap_unordered(
                        ReinforcementLearningFits.fit_softmax_policy_parallelizable, job_params):

                    mean_squared_error_sublist, optimized_params_sublist = task_result

                    mean_squared_error_list += mean_squared_error_sublist
                    optimized_params_list += optimized_params_sublist

                    progress_bar.update(step=progress_bar.i + len(mean_squared_error_sublist))

        else:
            single_job_params.append(1)  # append #fits = 1 (to be done by single job) to single_job_params

            for i in range(no_fits):
                # update progress_bar due to progress
                progress_bar.update(i)

                mean_squared_error_sublist, optimized_params_sublist = \
                    ReinforcementLearningFits.fit_softmax_policy_parallelizable(single_job_params)

                mean_squared_error_list += mean_squared_error_sublist
                optimized_params_list += optimized_params_sublist

        # finish progress bar
        progress_bar.finish()

        # plot best fit
        index = np.argmin(mean_squared_error_list)
        mean_squared_error_min = mean_squared_error_list[index]
        optimized_params_min = optimized_params_list[index]

        if no_fits > 1:
            mean_squared_error_2nd_smallest = np.partition(mean_squared_error_list, 1)[1]

        vals_fitted_func_array = softmax_policy(coords_array, optimized_params_min)

        if plot_diff:
            plot_prob_distribution(T, (data_array - vals_fitted_func_array),
                                   set_title=set_title,
                                   title=file_name + "_diff",
                                   plot_mask=plot_mask, diff=True)

        plot_prob_distribution(T, vals_fitted_func_array,
                               set_title=set_title,
                               title=file_name,
                               plot_mask=plot_mask, diff=False)

        # save plot parameters in txt-file
        policy_evaluation = PolicyEvaluation(T, vals_fitted_func_array, no_trajectories, s)

        write_plot_params_to_file(file_name, no_qubits, no_layers, no_fits, T, s, no_trajectories,
                                  mean_squared_error_list, policy_evaluation.prob_rare_trajectory,
                                  policy_evaluation.return_values_list, plot_name=file_name + "_P_to_go_1_step_down")

        return vals_fitted_func_array, optimized_params_min, mean_squared_error_list


    @staticmethod
    def fit_softmax_policy_parallelizable(job_params):
        """
        # TODO
        """
        # initializations
        no_thetas, no_amplitudes, no_phases, optimized_params_fourier_coeffs, optimized_no_layers, no_layers, \
            softmax_policy, coords_array, data_array, cost_func_type, no_trajectories, T, s, no_fits = job_params

        mean_squared_error_sublist = []
        optimized_params_sublist = []

        for fit in range(no_fits):
            if no_thetas is not None:
                initial_scalings = np.random.standard_normal(3)
                initial_thetas = 2 * np.pi * np.random.random(no_thetas)
                initial_params = np.insert(initial_scalings, 2, initial_thetas)
                # inserts initial_thetas into initial_scalings starting at position 2

                bounds_params = ([(-np.inf, np.inf)] * 2
                                 + [(0., 2 * np.pi)] * no_thetas
                                 + [(-np.inf, np.inf)])

            if no_amplitudes is not None and no_phases is not None:
                if optimized_params_fourier_coeffs is None or optimized_no_layers is None:
                    initial_scalings = np.random.standard_normal(3 + no_amplitudes)
                    initial_phases = 2 * np.pi * np.random.random(no_phases)
                    initial_params = np.insert(initial_scalings, 2 + no_amplitudes, initial_phases)  # FIXME
                    # inserts initial_phases into initial_scalings starting at position 2 + no_amplitudes

                else:
                    initial_params = np.zeros(2 + no_amplitudes + no_phases + 1)
                    initial_params = \
                        ReinforcementLearningFits.insert_optimized_params_into_larger_params_array(
                            optimized_params_fourier_coeffs,
                            initial_params,
                            optimized_no_layers,
                            no_layers)

                bounds_params = ([(-np.inf, np.inf)] * (2 + no_amplitudes)
                                 + [(0., 2 * np.pi)] * no_phases
                                 + [(-np.inf, np.inf)])

            optimized_params, mean_squared_error = \
                fit_func_to_data(softmax_policy, coords_array, data_array,
                                 params_initial_guess=initial_params, params_bounds=bounds_params,
                                 no_independent_vars=2, cost_func_type=cost_func_type, no_trajectories=no_trajectories,
                                 T=T, s=s)

            mean_squared_error_sublist.append(mean_squared_error)
            optimized_params_sublist.append(optimized_params)

        return mean_squared_error_sublist, optimized_params_sublist


class PolicyEvaluation:
    def __init__(self, T: int, policy_array: np.ndarray, no_trajectories: int, s: float,
                 reweighted_dynamics: ReweightedDynamics = None):
        # IDEA: implement this class to evaluate the policy policy_array by computing:
        # - average return of no_trajectories trajectories
        # - probability to generate a rare trajectory (random-walk bridge) for no_trajectories trajectories
        self.trajectories_x_array = self.calc_trajectories_x_array(no_trajectories, T, policy_array)

        self.return_values_list = self.calc_return_values(self.trajectories_x_array, T,
                                                          policy_array, s)

        self.average_return_estimate = np.mean(self.return_values_list)

        if reweighted_dynamics is not None:
            self.Kullback_Leibler_divergence_estimate = - self.average_return_estimate \
                                                        + reweighted_dynamics.partition_function_Z

        self.prob_rare_trajectory = self.calc_prob_rare_trajectory(self.trajectories_x_array, T)


    @staticmethod
    def generate_trajectory_with_policy(T: int, policy_array: np.ndarray):
        x = 0
        x_list = [x]

        for t in range(1, T + 1):
            prob_plus_1 = policy_array[t - 1, x_list[t - 1] + T - 1]
            delta_x = np.random.choice([+1, -1], p=[prob_plus_1, 1 - prob_plus_1])
            x = x + delta_x
            x_list.append(x)

        return x_list


    @staticmethod
    def calc_trajectories_x_array(no_trajectories: int, T: int, policy_array: np.ndarray):
        # initialize array
        trajectories_x_array = np.zeros((no_trajectories, T + 1), dtype=int)

        # compute array entries
        for n in range(no_trajectories):
            trajectories_x_array[n, :] = PolicyEvaluation.generate_trajectory_with_policy(T, policy_array)

        return trajectories_x_array


    @staticmethod
    def calc_return_values(trajectories_x_array: np.ndarray, T: int, policy_array: np.ndarray, s: float):
        # initializations and asserts
        no_trajectories, T_plus_1 = np.shape(trajectories_x_array)
        assert T_plus_1 == T + 1, "np.shape(trajectories_x_array) must be (positive int, T + 1)"

        p_distribution = np.where(np.isnan(policy_array), np.nan, 1/2)

        return_values_list = []

        # compute return values for trajectories
        for n in range(no_trajectories):
            return_value = 0

            for t in range(1, T + 1):
                return_value += ReweightedDynamics.calc_reward(trajectories_x_array[n, t],
                                                               trajectories_x_array[n, t - 1],
                                                               t, T, s, policy_array, p_distribution)

            return_values_list.append(return_value)

        return return_values_list


    @staticmethod
    def calc_prob_rare_trajectory(trajectories_x_array: np.ndarray, T: int):
        # initializations and asserts
        no_trajectories, T_plus_1 = np.shape(trajectories_x_array)
        assert T_plus_1 == T + 1, "np.shape(trajectories_x_array) must be (positive int, T + 1)"

        no_RWB = np.sum(trajectories_x_array[:, -1] == 0.)  # no of trajectories with endpoint x_T == 0.

        return no_RWB / no_trajectories


class AllPlotsFewQubitsCases:
    def __init__(self, fourier_coeffs: FourierCoeffs = None,
                 reinforcement_learning_fits: ReinforcementLearningFits | List[ReinforcementLearningFits] = None):
        """
        Utility class for few_qubits_cases.py which reproduces plots of classes FourierCoeffs and/or
        ReinforcementLearningFits based on the attributes of fourier_coeffs and/or reinforcement_learning_fits
        and moreover plots mean and variance of residual mean squared errors vs. no. of layers
        :param fourier_coeffs:
        :param reinforcement_learning_fits: single instance of class ReinforcementLearningFits or list of instances
        """
        self.plot_residual_mean_squared_errors_mean_std_vs_no_layers(reinforcement_learning_fits, set_title=True)


    @staticmethod
    def plot_residual_mean_squared_errors_mean_std_vs_no_layers(list_reinforcement_learning_fits:
                                                                List[ReinforcementLearningFits], set_title=False):
        # initialize lists
        list_no_layers = []
        list_mean_of_mean_squared_errors = []
        list_std_of_mean_squared_errors = []

        for n in range(len(list_reinforcement_learning_fits)):
            reinforcement_learning_fits = list_reinforcement_learning_fits[n]

            list_no_layers.append(reinforcement_learning_fits.no_layers)
            list_mean_of_mean_squared_errors.append(np.mean(
                reinforcement_learning_fits.residual_mean_squared_errors_1_qubit_fourier_coeffs))
            list_std_of_mean_squared_errors.append(np.std(
                reinforcement_learning_fits.residual_mean_squared_errors_1_qubit_fourier_coeffs))
            # TODO: implement further lists for further mean squared errors of interest

        fig, ax = plt.subplots()
        ax.errorbar(list_no_layers, list_mean_of_mean_squared_errors, yerr=list_std_of_mean_squared_errors,
                    fmt='o')

        if set_title:
            ax.set_title("statistics for residual mean squared errors of fits")

        # ax.set_ylim(-1, 1)
        # ax.set_xlim(-1, 1)

        plt.tight_layout(pad=0.5)

        # save plot
        fig.savefig("statistics_residual_mean_squared_errors_vs_no_layers.pdf",
                    bbox_inches="tight")

        plt.show()


class PlotTableResultsFewQubitsCases:
    def __init__(self, no_layers_list: list,
                 min_MSE_1_qubit_list: list, min_MSE_2_qubits: float,
                 mean_MSE_1_qubit_list: list, mean_MSE_2_qubits: float,
                 std_MSE_1_qubit_list: list, std_MSE_2_qubits: float,
                 prob_rare_trajectory_1_qubit_list: list, prob_rare_trajectory_2_qubits: float):
        # 3 different colors needed; ticks and labels of left and right axis different
        self.errorbar_plot(no_layers_list, min_MSE_1_qubit_list, min_MSE_2_qubits, mean_MSE_1_qubit_list,
                           mean_MSE_2_qubits, std_MSE_1_qubit_list, std_MSE_2_qubits, prob_rare_trajectory_1_qubit_list,
                           prob_rare_trajectory_2_qubits)


    @staticmethod
    def errorbar_plot(no_layers_list: list,
                      min_MSE_1_qubit_list: list, min_MSE_2_qubits: float,
                      mean_MSE_1_qubit_list: list, mean_MSE_2_qubits: float,
                      std_MSE_1_qubit_list: list, std_MSE_2_qubits: float,
                      prob_rare_trajectory_1_qubit_list: list, prob_rare_trajectory_2_qubits: float):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, width_ratios=[3, 1, 1])#, sharey=True)
        plt.subplots_adjust(wspace=0.15)  # adjusts width between subplots

        # plot first dataset with error bars
        color = 'tab:red'
        ax1.errorbar(no_layers_list, min_MSE_1_qubit_list, fmt='D', color=color, label='min(MSE)')
        ax1.errorbar(no_layers_list, mean_MSE_1_qubit_list, yerr=std_MSE_1_qubit_list, fmt='o', color=color,
                     label='$\mu(MSE) \pm \sigma(MSE)$')

        ax2.errorbar(no_layers_list, min_MSE_1_qubit_list, fmt='D', color=color, label='min(MSE)')
        ax2.errorbar(no_layers_list, mean_MSE_1_qubit_list, yerr=std_MSE_1_qubit_list, fmt='o', color=color,
                     label='$\mu(MSE) \pm \sigma(MSE)$')

        ax3.errorbar(no_layers_list, min_MSE_1_qubit_list, fmt='D', color=color, label='min(MSE)')
        ax3.errorbar(no_layers_list, mean_MSE_1_qubit_list, yerr=std_MSE_1_qubit_list, fmt='o', color=color,
                     label='$\mu(MSE) \pm \sigma(MSE)$')

        ax1.errorbar(1, min_MSE_2_qubits, fmt='D', mec=color, mfc='none')  # , label='min(MSE)')
        ax1.errorbar(1, mean_MSE_2_qubits, yerr=std_MSE_2_qubits, fmt='o', mec=color, mfc='none')  # ,
                     #label='$\mu(MSE) \pm \sigma(MSE)$')

        ax2.errorbar(1, min_MSE_2_qubits, fmt='D', mec=color, mfc='none')  # , label='min(MSE)')
        ax2.errorbar(1, mean_MSE_2_qubits, yerr=std_MSE_2_qubits, fmt='o', mec=color, mfc='none')  # ,
                     #label='$\mu(MSE) \pm \sigma(MSE)$')

        ax3.errorbar(1, min_MSE_2_qubits, fmt='D', mec=color, mfc='none')  # , label='min(MSE)')
        ax3.errorbar(1, mean_MSE_2_qubits, yerr=std_MSE_2_qubits, fmt='o', mec=color, mfc='none')  # ,
                     #label='$\mu(MSE) \pm \sigma(MSE)$')

        # create second y-axis sharing same x-axis
        ax4 = ax1.twinx()
        ax5 = ax2.twinx()
        ax6 = ax3.twinx()

        # plot second dataset with error bars
        color = 'tab:green'
        ax4.errorbar(no_layers_list, prob_rare_trajectory_1_qubit_list, fmt='s', color=color)
        ax4.errorbar(1, prob_rare_trajectory_2_qubits, fmt='s', mec=color, mfc='none')

        ax5.errorbar(no_layers_list, prob_rare_trajectory_1_qubit_list, fmt='s', color=color)
        ax5.errorbar(1, prob_rare_trajectory_2_qubits, fmt='s', mec=color, mfc='none')

        ax6.errorbar(no_layers_list, prob_rare_trajectory_1_qubit_list, fmt='s', color=color)
        ax6.errorbar(1, prob_rare_trajectory_2_qubits, fmt='s', mec=color, mfc='none')

        # adjust plots
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax6.spines['left'].set_visible(False)

        ax1.set_xlim(0., 6.)
        ax2.set_xlim(9., 11.)
        ax3.set_xlim(14., 16.)

        ax1.set_ylim(0.0, 1.01 * mean_MSE_2_qubits)
        ax2.set_ylim(0.0, 1.01 * mean_MSE_2_qubits)
        ax3.set_ylim(0.0, 1.01 * mean_MSE_2_qubits)
        ax4.set_ylim(0.0, 1.0)
        ax5.set_ylim(0.0, 1.0)
        ax6.set_ylim(0.0, 1.0)

        ax1.set_xticks([1, 2, 3, 4, 5])
        ax1.set_xticklabels([1, 2, 3, 4, 5])
        ax2.set_xticks([10])
        ax2.set_xticklabels([10])
        ax3.set_xticks([15])
        ax3.set_xticklabels([15])

        tick_step = 1.01 * mean_MSE_2_qubits / 5.
        ax1.set_yticks([0., tick_step, 2 * tick_step, 3 * tick_step, 4 * tick_step, 5 * tick_step])
        ax1.set_yticklabels(
            [f"{x:.2f}" for x in [0., tick_step, 2 * tick_step, 3 * tick_step, 4 * tick_step, 5 * tick_step]])
        ax1.tick_params(axis='y', which='both', labelleft=True, left=True)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_yticklabels([])

        ax4.set_yticks([])
        ax4.set_yticklabels([])
        ax5.set_yticks([])
        ax5.set_yticklabels([])

        color = 'tab:red'
        ax1.set_ylabel('mean squared error (MSE)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:green'
        ax6.set_ylabel('prob. rare trajectory $P(x_T = 0)$', color=color)
        ax6.tick_params(axis='y', labelcolor=color)

        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('# data-uploading layers')

        #ax2.legend(loc='upper right', fontsize=12)  # , labels=['Quantity 1', 'Quantity 2'])

        fig.tight_layout()

        fig.savefig("plot_table_results_few_qubits_cases.pdf", bbox_inches="tight")

        plt.show()
