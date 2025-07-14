"""
MIT License
Copyright © 2025 David A. Reiss
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
import argparse
import copy
import json
import math
import warnings
import json5
import pytest
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any
from pathlib import Path

main_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(main_directory))

from config_template import Config
from reweighted_dynamics import ReweightedDynamics
from Fourier_series_analysis_and_fits import ParameterizedDynamicsFits
from utilities import save_obj, prepare_results_dir, get_file_names_with_version, InfoMessage
from benchmark_utilities import BenchmarkUtilities


# tests
@pytest.mark.parametrize("T", BenchmarkUtilities._get_config_benchmark("config_reweighted_dynamics_scaling_time_horizon.json5")["T"])
def test_reweighted_dynamics_scaling_time_horizon(T, config_reweighted_dynamics_benchmark, benchmark):
    config = BenchmarkUtilities._get_config(config_reweighted_dynamics_benchmark, T)

    benchmark(lambda: BenchmarkUtilities._compute_reweighted_dynamics(config))


@pytest.mark.parametrize("T", BenchmarkUtilities._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")["T"])
def test_parameterized_dynamics_fits_scaling_time_horizon(T, config_parameterized_dynamics_fits_benchmark,
                                                          required_min_residual_cost, benchmark,
                                                          save_result=True):
    """
    Benchmark computation time for fitting parameterized dynamics to reweighted dynamics P_W.

    Parameters:
        T: time horizon of random walk
        config_benchmark: configuration for benchmark, provided as a fixture
        required_min_residual_cost: minimum residual cost for fits, provided as a fixture
        benchmark: pytest-benchmark fixture to measure time taken for computation
        save_result: whether to save result of fits to a file;
            NOTE that if True this might slightly distort the benchmark time
    """

    # TODO: clarify whether different seeds for np.random should be used for each T value

    benchmark_utils = BenchmarkUtilities

    config = benchmark_utils._get_config(config_parameterized_dynamics_fits_benchmark, T)

    reweighted_dynamics = benchmark_utils._compute_reweighted_dynamics(config)

    path_computations, path_plots = prepare_results_dir(f"config_parameterized_dynamics_fits_scaling_time_horizon_{T}.json5",
                                                        dump_dict=config.__dict__)

    Path(path_plots).rmdir()

    if config.fitting_parameters in ("Fourier_coefficients", "variational_angles"):
        file_name_list = [f"{path_computations}/fits_qubits_{config.no_qubits_list[0]}_layers_{config.no_layers_list[0]}"
                          f"_{config.cost_func_type}_fitting_parameters_{config.fitting_parameters}.npz"]

    if config.fitting_parameters == "random_Fourier_features":
        file_name = (f"{path_computations}/fits_qubits_{config.no_qubits_list[0]}_layers_{config.no_layers_list[0]}"
                     f"_{config.cost_func_type}_random_Fourier_features_{config.no_random_Fourier_features}.npz")

        file_name_list, _ = get_file_names_with_version(file_name, config.no_choices_random_Fourier_features,
                                                        path_computations)

    benchmark(lambda: benchmark_utils._fit_parameterized_dynamics_to_required_accuracy(file_name_list, reweighted_dynamics,
                                                                                       config, required_min_residual_cost,
                                                                                       save_result,
                                                                                       compute_in_parallel=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling benchmarks with optional specification of benchmark.")
    parser.add_argument("--benchmarks", type=str, default="all",
                        help="Benchmarks to be executed (default: 'all'; options: 'all', "
                             "'test_reweighted_dynamics_scaling_time_horizon', "
                             "'test_parameterized_dynamics_fits_scaling_time_horizon')")
    benchmarks = parser.parse_args().benchmarks
    benchmark_utils = BenchmarkUtilities


    # run scaling benchmarks and plot results if this script is executed directly
    if benchmarks == "all" or benchmarks == "test_reweighted_dynamics_scaling_time_horizon":
        print("Running benchmark for reweighted dynamics scaling with time horizon...")

        pytest.main([__file__,
                     "-k", "test_reweighted_dynamics_scaling_time_horizon",
                     "--benchmark-enable", "--benchmark-only",
                     "--benchmark-json=data_reweighted_dynamics_scaling_time_horizon.json",
                     "--benchmark-min-rounds=2"])

        df = benchmark_utils._load_results('data_reweighted_dynamics_scaling_time_horizon.json')
        benchmark_utils._plot_results(df, "plot_reweighted_dynamics_scaling_time_horizon.pdf",
                                      r"$\text{Scaling benchmark for computation of reweighted dynamics}~P_W$")


    if benchmarks == "all" or benchmarks == "test_parameterized_dynamics_fits_scaling_time_horizon":
        print("Running benchmark for parameterized dynamics fits scaling with time horizon...")

        pytest.main([__file__,
                     "-k", "test_parameterized_dynamics_fits_scaling_time_horizon",
                     "--benchmark-enable", "--benchmark-only",
                     "--benchmark-json=data_parameterized_dynamics_fits_scaling_time_horizon.json"])

        df = benchmark_utils._load_results('data_parameterized_dynamics_fits_scaling_time_horizon.json')
        config_benchmark = benchmark_utils._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")
        no_qubits = config_benchmark["no_qubits_list"][0]
        no_layers = config_benchmark["no_layers_list"][0]
        pqc_str = r"~\text{for " + f"{no_qubits} qubits, {no_layers} layers" + r"}$"
        benchmark_utils._plot_results(df, "plot_parameterized_dynamics_fits_scaling_time_horizon.pdf",
                                      r"$\text{Scaling benchmark for computation of fits of parameterized dynamics}~P_\theta~\text{to}~P_W"
                                      + pqc_str)
