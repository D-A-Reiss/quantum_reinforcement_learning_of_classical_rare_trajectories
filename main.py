from reweighted_dynamics import ReweightedDynamics
from few_qubits_cases import ReinforcementLearningFits, FourierCoeffs, AllPlotsFewQubitsCases, PolicyEvaluation, \
    PlotTableResultsFewQubitsCases
from utilities import restore_or_compute_obj, load_and_restore_obj

if __name__ == '__main__':
    # parameters of computations for few-qubit cases
    T = 20
    s = 1
    x_0 = 0.
    prob_step_up = 0.5

    no_qubits = 1  # 1
    no_samples = 10000  # 4 ** 8  # 4 ** 7  # 10 ** 3  # 100  # 1000

    no_layers = 2  # 15
    no_fits = 100
    no_trajectories = 100000
    set_title = False
    recompute = True

    # computations for few-qubit cases
    reweighted_dynamics = restore_or_compute_obj(ReweightedDynamics,
                                                 lambda: ReweightedDynamics(T, s, x_0=x_0, prob_step_up=prob_step_up,
                                                                            calc_value_function=False),
                                                 "reweighted_dynamics.npz", recompute=False)

    """
    reinforcement_learning_fits = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           no_trajectories,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits.npz",
                                                         recompute=False)

    FourierCoeffs(no_qubits, no_layers, no_samples, random_thetas=True,
                  optimized_fourier_coeffs=reinforcement_learning_fits.optimized_params_1_qubit_fourier_coeffs[2:-1])
    """

    """
    reweighted_dynamics = restore_or_compute_obj(ReweightedDynamics,
                                                 lambda: ReweightedDynamics(T, s, x_0=x_0, prob_step_up=prob_step_up,
                                                                            calc_value_function=False),
                                                 "reweighted_dynamics.npz", recompute=False)
    no_fits = 1
    reinforcement_learning_fits = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           no_trajectories,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits.npz",
                                                         recompute=False)

    print(reinforcement_learning_fits.residual_mean_squared_errors_1_qubit_fourier_coeffs)

    no_fits = 1
    reinforcement_learning_fits_more_layers = restore_or_compute_obj(ReinforcementLearningFits,
                                                                     lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                                       no_layers + 1,
                                                                                                       no_fits,
                                                                                                       no_trajectories,
                                                                                                       set_title=set_title,
                                                                                                       theta_fits=False,
                                                                                                       optimized_params_fourier_coeffs=reinforcement_learning_fits.optimized_params_1_qubit_fourier_coeffs,
                                                                                                       optimized_no_layers=no_layers),
                                                                     "reinforcement_learning_few_qubits_" + str(no_layers + 1) +
                                                                     "_layers_" + str(no_fits) + "_fits.npz",
                                                                     recompute=True)

    print(reinforcement_learning_fits_more_layers.residual_mean_squared_errors_1_qubit_fourier_coeffs)
    """

    evaluation_of_P_W = restore_or_compute_obj(PolicyEvaluation,
                                               lambda: PolicyEvaluation(T, reweighted_dynamics.P_W_array,
                                                                        no_trajectories, s),
                                               "evaluation_of_P_W.npz",
                                               recompute=recompute)

    print(evaluation_of_P_W.prob_rare_trajectory)
    print(evaluation_of_P_W.average_return_estimate)
    print(evaluation_of_P_W.Kullback_Leibler_divergence_estimate)

    """
    no_layers_list = [1, 2, 3, 4, 5, 10, 15]
    min_MSE_1_qubit_list = [2.76e-3, 1.14e-3, 0.80e-3, 0.87e-3, 1.13e-3, 5.67e-3, 11.8e-3]
    min_MSE_2_qubits = 1.49e-1
    mean_MSE_1_qubit_list = [4.76e-3, 3.19e-3, 3.99e-3, 3.67e-3, 5.82e-3, 28.1e-3, 78.6e-3]
    mean_MSE_2_qubits = 1.49e-1
    std_MSE_1_qubit_list = [3.60e-3, 6.27e-3, 8.66e-3, 7.20e-3, 9.40e-3, 27.7e-3, 66.6e-3]
    std_MSE_2_qubits = 0.
    prob_rare_trajectory_1_qubit_list = [0.59, 0.71, 0.76, 0.73, 0.73, 0.64, 0.47]
    prob_rare_trajectory_2_qubits = 0.18

    PlotTableResultsFewQubitsCases(no_layers_list, min_MSE_1_qubit_list, min_MSE_2_qubits, mean_MSE_1_qubit_list,
                                   mean_MSE_2_qubits, std_MSE_1_qubit_list, std_MSE_2_qubits,
                                   prob_rare_trajectory_1_qubit_list, prob_rare_trajectory_2_qubits)
    """

