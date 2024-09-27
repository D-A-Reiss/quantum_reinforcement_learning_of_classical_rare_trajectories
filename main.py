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
    # only relevant to computations in class FourierCoeffs
    no_samples = 10000  # 4 ** 8  # 4 ** 7  # 10 ** 3  # 100  # 1000
    # specifies #sets of randomly chosen variational angles = #times Fourier coefficients are computed

    no_layers = 2  # 15
    no_fits = 100
    no_trajectories = 1000  # 100 in combination with 1 layers shows the following interesting effect:
    # P_theta_1_qubit.prob_rare_trajectory:  0.417
    # P_theta_1_qubit.average_return_estimate:  -4.676786856728079
    # > quite high average_return_estimate and (for that) quite low prob_rare_trajectory
    set_title = False
    recompute = True

    cost_func_type = "trajectory_KL_divergence"

    # computations for few-qubit cases
    # compute reweighted dynamics/optimal policy P_W
    reweighted_dynamics = restore_or_compute_obj(ReweightedDynamics,
                                                 lambda: ReweightedDynamics(T, s, x_0=x_0, prob_step_up=prob_step_up,
                                                                            calc_value_function=False),
                                                 "reweighted_dynamics.npz", recompute=False)

    no_trajectories = 100000
    # heuristic value where estimates of prob. to generate rare trajectory and of average/expected return
    # seem to have converged

    evaluation_of_P_W = restore_or_compute_obj(PolicyEvaluation,
                                               lambda: PolicyEvaluation(T, reweighted_dynamics.P_W_array,
                                                                        no_trajectories, s),
                                               "evaluation_of_P_W.npz",
                                               recompute=False)

    print("P_W.prob_rare_trajectory: ", evaluation_of_P_W.prob_rare_trajectory)
    print("P_W.average_return_estimate: ", evaluation_of_P_W.average_return_estimate)

    # plot results in table comparing different parameter cases
    """
    #no_layers_list = [1, 2, 3, 4, 5, 10, 15]
    no_layers_list = [2, 3, 4, 5, 10, 15]
    
    #min_MSE_1_qubit_list = [2.76e-3, 1.14e-3, 0.80e-3, 0.87e-3, 1.13e-3, 5.67e-3, 11.8e-3]
    #min_MSE_2_qubits = 1.49e-1
    #mean_MSE_1_qubit_list = [4.76e-3, 3.19e-3, 3.99e-3, 3.67e-3, 5.82e-3, 28.1e-3, 78.6e-3]
    #mean_MSE_2_qubits = 1.49e-1
    #std_MSE_1_qubit_list = [3.60e-3, 6.27e-3, 8.66e-3, 7.20e-3, 9.40e-3, 27.7e-3, 66.6e-3]
    #std_MSE_2_qubits = 0.

    fits_2_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_2_layers_100_fits.npz")
    fits_3_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_3_layers_100_fits.npz")
    fits_4_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_4_layers_100_fits.npz")
    fits_5_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_5_layers_100_fits.npz")
    
    fits_10_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_10_layers_100_fits.npz")
    
    fits_15_layers = load_and_restore_obj(ReinforcementLearningFits, 
                                         "reinforcement_learning_few_qubits_15_layers_100_fits.npz")
    
    list_policies = [load_and_restore_obj(ReinforcementLearningFits, 
                                         f"reinforcement_learning_few_qubits_{no_layers}_layers_100_fits.npz").vals_fitted_func_1_qubit_fourier_coeffs 
                     for no_layers in [2, 3, 4, 5, 10, 15]]
    
    evaluation_policies = [PolicyEvaluation(T, policy, no_trajectories, s, 
                                            average_return_estimate_P_W=evaluation_of_P_W.average_return_estimate)
                           for policy in list_policies]

    min_KL_div_1_qubit_list =
    min_KL_div_2_qubits =
    mean_KL_div_1_qubit_list =
    mean_KL_div_2_qubits =
    std_KL_div_1_qubit_list =
    std_KL_div_2_qubits =
    prob_rare_trajectory_1_qubit_list = [0.59, 0.71, 0.76, 0.73, 0.73, 0.64, 0.47]
    prob_rare_trajectory_2_qubits = 0.18
    diff_prob_rare_trajectory_1_qubit_list = [evaluation_of_P_W.prob_rare_trajectory - prob
                                              for prob in prob_rare_trajectory_1_qubit_list]
    diff_prob_rare_trajectory_2_qubits = evaluation_of_P_W.prob_rare_trajectory - 0.18

    PlotTableResultsFewQubitsCases(no_layers_list, min_KL_div_1_qubit_list, min_KL_div_2_qubits, mean_KL_div_1_qubit_list,
                                   mean_KL_div_2_qubits, std_KL_div_1_qubit_list, std_KL_div_2_qubits,
                                   diff_prob_rare_trajectory_1_qubit_list, diff_prob_rare_trajectory_2_qubits)
    """

    """
    # compute Fourier coefficients of parameterized dynamics
    FourierCoeffs(no_qubits, no_layers, no_samples, random_thetas=True,
                  optimized_fourier_coeffs=reinforcement_learning_fits.optimized_params_1_qubit_fourier_coeffs[2:-1])
    """

    # fit dynamics parameterized by Fourier coefficients to P_W
    reinforcement_learning_fits = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           no_trajectories,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False,
                                                                                           compute_in_parallel=True,
                                                                                           cost_func_type=cost_func_type),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits_" + cost_func_type + ".npz",
                                                         recompute=recompute)

    """
    # fit dynamics parameterized by Fourier coefficients to P_W SUCCESSIVELY
    # by repeatedly adding one data-uploading layer and 
    # reusing optimized Fourier coefficients for one layer less as starting point for fits
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

    # evaluation of properties of P_W and parameterized dynamics
    no_trajectories = 100000
    # heuristic value where estimates of prob. to generate rare trajectory and of average/expected return
    # seem to have converged

    evaluation_of_P_W = restore_or_compute_obj(PolicyEvaluation,
                                               lambda: PolicyEvaluation(T, reweighted_dynamics.P_W_array,
                                                                        no_trajectories, s),
                                               "evaluation_of_P_W.npz",
                                               recompute=False)

    print("P_W.prob_rare_trajectory: ", evaluation_of_P_W.prob_rare_trajectory)
    print("P_W.average_return_estimate: ", evaluation_of_P_W.average_return_estimate)
    #print(evaluation_of_P_W.Kullback_Leibler_divergence_estimate)

    evaluation_of_P_theta = restore_or_compute_obj(PolicyEvaluation,
                                               lambda: PolicyEvaluation(T, reinforcement_learning_fits.vals_fitted_func_1_qubit_fourier_coeffs,
                                                                        no_trajectories, s),
                                               "evaluation_of_P_theta_1_qubit_" + str(no_layers) + "_layers_"
                                                   + str(no_fits) + "_fits_" + cost_func_type +".npz",
                                               recompute=recompute)

    print("P_theta_1_qubit.prob_rare_trajectory: ", evaluation_of_P_theta.prob_rare_trajectory)
    print("P_theta_1_qubit.average_return_estimate: ", evaluation_of_P_theta.average_return_estimate)

    if no_layers == 1:
        evaluation_of_P_theta = restore_or_compute_obj(PolicyEvaluation,
                                                       lambda: PolicyEvaluation(T,
                                                                                reinforcement_learning_fits.vals_fitted_func_2_qubits_fourier_coeffs,
                                                                                no_trajectories, s),
                                                       "evaluation_of_P_theta_2_qubits_" + str(no_layers) + "_layers_"
                                                       + str(no_fits) + "_fits_" + cost_func_type + ".npz",
                                                       recompute=recompute)

        print("P_theta_2_qubits.prob_rare_trajectory: ", evaluation_of_P_theta.prob_rare_trajectory)
        print("P_theta_2_qubits.average_return_estimate: ", evaluation_of_P_theta.average_return_estimate)

