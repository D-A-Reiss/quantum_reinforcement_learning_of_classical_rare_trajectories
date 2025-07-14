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
import pytest
import sys
import numpy as np
from pathlib import Path

main_directory = Path(__file__).resolve().parent.parent
sys.path.append(str(main_directory))

from config_template import Config
from benchmark_utilities import BenchmarkUtilities


def pytest_addoption(parser):
    parser.addoption("--T", action="store", default=None, help="Value for x")


@pytest.fixture
def T(request):
    return request.config.getoption("--T")


@pytest.fixture(scope="session")
def config_reweighted_dynamics_benchmark():
    return BenchmarkUtilities._get_config_benchmark("config_reweighted_dynamics_scaling_time_horizon.json5")


@pytest.fixture(scope="session")
def config_parameterized_dynamics_fits_benchmark():
    return BenchmarkUtilities._get_config_benchmark("config_parameterized_dynamics_fits_scaling_time_horizon.json5")


@pytest.fixture(scope="session")
def required_min_residual_cost(config_parameterized_dynamics_fits_benchmark):
    config = BenchmarkUtilities._get_config(config_parameterized_dynamics_fits_benchmark,
                                            config_parameterized_dynamics_fits_benchmark["T"][0])
    reweighted_dynamics = BenchmarkUtilities._compute_reweighted_dynamics(config)

    # set required min. residual cost to min. residual cost for first T value and no_fits_first_T_value-many fits
    config.no_fits = config_parameterized_dynamics_fits_benchmark["no_fits_first_T_value"]
    min_residual_cost = BenchmarkUtilities._fit_parameterized_dynamics_to_required_accuracy(
        ["mock_name"], reweighted_dynamics, config, np.inf, save_result=False, compute_in_parallel=True)
    return min_residual_cost