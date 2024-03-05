from reweighted_dynamics import ReweightedDynamics
from few_qubits_cases import ReinforcementLearningFits, FourierCoeffs, AllPlotsFewQubitsCases, PolicyEvaluation
from utilities import restore_or_compute_obj, load_and_restore_obj

if __name__ == '__main__':
    # calculations for 1-qubit case with
    T = 20
    s = 1
    no_qubits = 2  # 1
    no_layers = 1  # 3
    no_random_samples = 100  # 100
    no_fits = 10
    set_title = False


    FourierCoeffs(no_qubits, no_layers, no_random_samples)
    """
    reweighted_dynamics = restore_or_compute_obj(ReweightedDynamics, lambda: ReweightedDynamics(T, s),
                                                 "reweighted_dynamics.npz")
    """
    """
    reinforcement_learning_fits = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           set_title=set_title,
                                                                                           theta_fits=True),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits.npz")

    # TODO: correct naming such that it becomes consistent
    """
    """
    no_qubits = 1
    no_fits = 100

    no_layers = 1
    reinforcement_learning_fits_1 = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits_no_thetas.npz")

    no_layers = 2
    reinforcement_learning_fits_2 = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits_no_thetas.npz")

    no_layers = 3
    reinforcement_learning_fits_3 = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits_no_thetas.npz")

    no_layers = 5
    reinforcement_learning_fits_5 = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits_no_thetas.npz")

    AllPlotsFewQubitsCases(reinforcement_learning_fits=[reinforcement_learning_fits_1, reinforcement_learning_fits_2,
                                                        reinforcement_learning_fits_3, reinforcement_learning_fits_5])

    no_trajectories = 100000

    PolicyEvaluation(T, reinforcement_learning_fits_1.vals_fitted_func_1_qubit_fourier_coeffs, no_trajectories, s)
    """





