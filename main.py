from reweighted_dynamics import ReweightedDynamics
from few_qubits_cases import ReinforcementLearningFits, FourierCoeffs, AllPlotsFewQubitsCases, PolicyEvaluation
from utilities import restore_or_compute_obj, load_and_restore_obj

if __name__ == '__main__':
    # parameters of computations for few-qubit cases
    T = 20
    s = 1

    no_qubits = 1  # 1
    no_random_samples = 100  # 100

    no_layers = 1  # 10, 15
    no_fits = 100
    no_trajectories = 100000
    set_title = False
    recompute = True

    # computations for few-qubit cases

    # FourierCoeffs(no_qubits, no_layers, no_random_samples)

    reweighted_dynamics = restore_or_compute_obj(ReweightedDynamics, lambda: ReweightedDynamics(T, s),
                                                 "reweighted_dynamics.npz")

    reinforcement_learning_fits = restore_or_compute_obj(ReinforcementLearningFits,
                                                         lambda: ReinforcementLearningFits(reweighted_dynamics, T,
                                                                                           no_layers, no_fits,
                                                                                           no_trajectories,
                                                                                           set_title=set_title,
                                                                                           theta_fits=False),
                                                         "reinforcement_learning_few_qubits_" + str(no_layers) +
                                                         "_layers_" + str(no_fits) + "_fits.npz",
                                                         recompute=recompute)
