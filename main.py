from one_qubit_case import OptimalPolicyCalculations, FourierCoeffs

if __name__ == '__main__':
    # calculations for 1-qubit case with
    T = 20
    s = 1

    no_layers = 1
    # no_layers = 2

    OptimalPolicyCalculations(T, s, no_layers)
    # FourierCoeffs(no_layers)



