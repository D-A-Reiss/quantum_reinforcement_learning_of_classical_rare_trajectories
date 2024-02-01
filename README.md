# QRL_LDT_work_in_progress

TO-DO-s
- browse literature whether there's a theorem which states that one can reach all possible 1-qubit states with (the given) 3 rotations on the Bloch sphere (catchword: Euler angles)
- https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.OneQubitEulerDecomposer might help
- try to adapt the proofs and arguments in the paper by Schuld, Sweke, and Meyer regarding the quantum models EXPRESSIVITY regarding the Fourier COEFFICIENTS of the truncated Fourier series

TO-DO-S for Laura's code:
- check whether numerical results for 1-qubit case do NOT depend on variational parameter theta_4
- to this end either implement function which can take all variational parameters and returns corresponding policy
- or to this end run existing algorithm without learning and check resulting policy for different initializations of theta_4

I'd propose the second option, so:
- implement that initialization of trainable parameters can also be NOT only random, but given by the user via the "init files"
- check whether it runs if variable "trainable" is always set to false and if so, also implement that this is determined by the user via the "init files"
- if both points above are done, run the code several times for the smallest possible number of learning episodes with always the same random initialization of trainable parameters except theta_4, which should be varied each time
