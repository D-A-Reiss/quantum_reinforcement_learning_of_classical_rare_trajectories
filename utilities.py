"""
MIT License
Copyright © 2024 David A. Reiss
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


import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import ABC, abstractmethod
from logging_config import get_logger


logger = get_logger("utilities.py")


def import_params_from_csv(path: str) -> dict:
    """
    Import parameters from a CSV file and return them as a dictionary.

    Parameters:
        path: path to the CSV file

    Returns:
        params: dict with parameter names and values as dict key-value pairs
    """

    params = {}

    with open(path, mode="r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            key, value = row["Parameter"], row["Value"]

            # convert types appropriately
            if value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                params[key] = float(value)
            elif value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            else:
                params[key] = value

    return params


def write_plot_related_params_to_csv_file(plot_params: dict, path_default_set_params_csv_file: str,
                                          path_new_csv_file: str) -> None:
    # TODO: maybe replace this function by one that creates an HTML file with a table of the parameters and their values
    #   alongside the representative plot
    """
    Write parameters related to a plot into a new CSV file.

    Parameters:
        plot_params: dict of parameter-value pairs related to plot
        path_default_set_params_csv_file: path to CSV file with default set of parameters for that kind of plot
        path_new_csv_file: path to new CSV file to be created
    Returns:
        None
    """

    # read CSV file to get default set of parameters
    default_set_params = import_params_from_csv(path_default_set_params_csv_file)

    # assert that plot_params conforms to default set of parameters
    assert plot_params.keys() == default_set_params.keys(), \
        "plot_params does not conform to CSV file path_default_set_params_csv_file"

    # write plot_params to new CSV file
    with open(path_default_set_params_csv_file, mode='r') as default_file, \
            open(path_new_csv_file, mode='w', newline='') as new_file:
        reader = csv.DictReader(default_file)

        writer = csv.DictWriter(new_file, fieldnames=["Parameter", "Value"])
        writer.writeheader()

        for row in reader:
            row["Value"] = plot_params[row["Parameter"]]
            writer.writerow(row)


def plot_prob_distribution(T: int, prob_array: np.ndarray, colorbar_label: str, set_title=True, title="",
                           plot_mask: np.ndarray = None, plot_complement=False, save_fig_as: str = None) -> None:
    """
    Plot probability distribution/policy as function of t and x in form of a heat map and save it as PDF.

    Parameters:
        T: maximal time
        prob_array: array of probabilities to be plotted
        colorbar_label: label of the color bar
        set_title: if True, set title of plot
        title: title of plot
        plot_mask: mask for plotting (if None, plot all values of prob_array)
            (e.g., for consistency with plot of P_W,
            plot_mask = np.isnan(reweighted_dynamics.P_W_array) with reweighted_dynamics: ReweightedDynamics)
        plot_complement: if True, plot 1 - prob_array
        save_fig_as: if not None, save plot as PDF with name save_fig_as

    Returns:
        None
    """

    # prepare prob_array for plotting with imshow
    if plot_mask is not None:
        prob_array = np.where(plot_mask, np.nan, prob_array)

    prob_array = np.swapaxes(prob_array, 0, 1)
    # now indices x,t

    if plot_complement:
        prob_array = 1 - prob_array

    # plot as heat map
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    if plot_complement:
        im = ax.imshow(prob_array, cmap='viridis', vmin=0., vmax=1.)
    else:
        im = ax.imshow(prob_array, cmap='viridis')

    # set color bar, title, labels, ticks, and limits
    fig.colorbar(im, cax=cax, orientation='vertical', label=colorbar_label)

    if set_title:
        ax.set_title(title)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    ax.set_xticks(np.array([0, 1 * T // 4, 2 * T // 4, 3 * T // 4, T]),
                  labels=[str(0), str(1 * T // 4), str(2 * T // 4), str(3 * T // 4), str(T)])

    ax.set_ylim([1 / 2 * T - 2, 3 / 2 * T])
    ax.set_yticks(np.array([2 * T // 4 - 1, 3 * T // 4 - 1, T - 1, 5 * T // 4 - 1, 6 * T // 4 - 1]),
                  labels=[str(-2 * T // 4), str(-1 * T // 4), str(0), str(1 * T // 4), str(2 * T // 4)])

    # save plot
    if save_fig_as is not None:
        fig.savefig(save_fig_as, bbox_inches="tight")

    plt.show()


def save_obj(obj, file_name: str) -> None:
    """
    Save all attributes of instance obj of a class to a file with name file_name.

    Parameters:
        obj: instance of a class
        file_name: string ending with '.npz'

    Returns:
        None
    """

    attributes = obj.__dict__
    np.savez(file_name, **attributes)


def load_and_restore_obj(class_type, file_name: str, all_params_dict: dict):
    """
    Load all attributes of an instance of class cls from a file with name file_name,
    create a new instance of class cls, and update its attributes accordingly.

    Parameters:
        class_type: class
        file_name: string ending with '.npz'

    Returns:
        obj: instance of class class_type
    """

    data = np.load(file_name, allow_pickle=True)

    obj = class_type.__new__(class_type)

    attributes = {f: data[f] for f in data.files}
    obj.__dict__.update(attributes)

    if not isinstance(obj, ConsistentParametersClass):
        error_message = (f"Class {class_type} must inherit from class ConsistentParametersClass to ensure that its "
                         f"objects can be restored with consistent parameters.")

        logger.debug(error_message)
        raise AssertionError(error_message)

    if not isinstance(obj.all_init_params_dict, dict):
        error_message = (f"Property all_init_params_dict of class {class_type} must be a dictionary of all "
                         f"parameter-value pairs used to initialize an instance.")

        logger.debug(error_message)
        raise AssertionError(error_message)

    if not all(item in all_params_dict.items() for item in obj.all_init_params_dict.items()):
        error_message = ("All parameter-value pairs in all_init_params_dict of loaded object must be consistent with "
                         "parameter-value pairs in all_params_dict which contains all parameter-value pairs used for "
                         "current computations.")

        logger.debug(error_message)
        raise AssertionError(error_message)

    return obj


def load_or_compute_obj(class_type, generator_function, file_name: str, all_params_dict: dict, recompute=False):
    """
    Load and restore an object of class class_type from a file file_name if it exists,
    otherwise compute it and save it to the file.

    Parameters:
        class_type: class
        generator_function: function that generates the object
        file_name: string ending with '.npz'
        recompute: if True, recompute the object even if it exists in the file and overwrite the file

    Returns:
        obj: instance of class class_type
    """

    if not recompute:
        try:
            obj = load_and_restore_obj(class_type, file_name, all_params_dict)
            logger.info(f"Loaded object of class {class_type} from {file_name}.")

            return obj

        except FileNotFoundError:
            logger.info(f"File {file_name} not found.")

        except AssertionError:
            logger.info(f"Loaded object of {class_type} from {file_name} cannot be restored consistently.")

    # so if recompute is True or if load_and_restore_obj failed due to file with file_name not existing
    # or due to all_init_params_dict of loaded object not consistent with all_params_dict used for current computations:
    # recompute the object and save it to the file

    obj = generator_function()
    logger.info(f"Computed object of {class_type}.")

    save_obj(obj, file_name)
    logger.info(f"Saved object of {class_type} in {file_name}.")

    return obj


def einsum_subscripts(*initial_indices: str, final_indices="") -> str:
    """
    Convert strings of indices in *initial_indices and string of indices in **final_indices to one string suitable as
    parameter "subscript" of numpy.einsum (see its documentation for details). (The strings can contain numbers, primes,
    superscripts, etc. like in common physicists' notation, unlike numpy.einsum subscripts.)

    Parameters:
        *initial_indices: strings of indices for operands of numpy.einsum
        final_indices: string of indices for result of numpy.einsum

    Returns:
        translated_indices: string of indices suitable as parameter "subscript" of numpy.einsum
    """

    # initializations
    letters_list = list(map(chr, range(97, 123)))  # list of all letters in the alphabet
    indices_dict = {}  # dictionary which contains to correspondences between subscripts in SB21/my notes
                         # and subscripts conforming with the conventions/requirements of np.einsum
    initial_indices_lists = [index.split(",") for index in initial_indices]
    final_indices_list = final_indices.split(",")
    translated_indices = ""

    # utility function
    def translate_and_add(index, translated_indices, indices_dict):
        # translate index according to indices_dict and add it to string translated_indices
        if index == "..." or index == "":
            translated_indices += index

        else:
            translated_indices += indices_dict[index]

        return translated_indices

    # translate indices
    for indices_list in initial_indices_lists:
        for index in indices_list:
            if index not in indices_dict.keys():
                # add index to indices_dict
                l = len(indices_dict)
                indices_dict.update({index: letters_list[l]})

            translated_indices = translate_and_add(index, translated_indices, indices_dict)

        translated_indices += ","  # in np.einsum, indices of different operands are to be separated by commas

    translated_indices = translated_indices[:-1]  # remove last comma
    translated_indices += "->"

    for index in final_indices_list:
        translated_indices = translate_and_add(index, translated_indices, indices_dict)

    return translated_indices


class ConsistentParametersClass(ABC):
    """
    Abstract base class for classes for which one wants to use function load_and_restore_obj (for example via function
    load_or_compute_obj) to load and restore instances of the respective class from files. The combination of this class
    and this function ensures that the parameters which had been used to initialize the now restored instance are
    CONSISTENT with the parameters which are used in the CURRENT computations.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def all_init_params_dict(self):
        """Dictionary of all parameter-value pairs used to initialize the respective object."""
        pass


class ProgressBar:
    """
    Class to create and update a progress bar for a task with a given number of steps
    """

    def __init__(self, total_no_steps: int | None = None, task_description: str = ""):
        """
        Initialize progress bar by setting attributes.

        Parameters:
            total_no_steps: total number of steps
            task_description: description of the task to be performed
        """

        self.start_time = time.time()
        self.current_step = -1
        self.total_no_steps = total_no_steps
        self.task_description = task_description


    @staticmethod
    def format_delta_time(time_in_seconds: float) -> str:
        """
        Format time in seconds as hours, minutes, and seconds.

        Parameters:
            time_in_seconds: time in seconds

        Returns:
            formatted time string
        """

        seconds = time_in_seconds % 60
        minutes = int((time_in_seconds // 60) % 60)
        hours = int((time_in_seconds // 60 // 24))

        if hours:
            return f"{hours} h {minutes} min"
        elif minutes:
            return f"{minutes} min {seconds:0.1f} s"
        else:
            return f"{seconds:0.1f} s"


    def update(self, new_current_step: int | None = None, new_total_no_steps: int | None = None):
        """
        Update and print updated progress bar.

        Parameters:
            new_current_step: new current step (if None, increment current_step by 1)
            new_total_no_steps: new total number of steps (if None, keep total_no_steps)

        Returns:
            None
        """

        # update total_no_steps if necessary
        if new_total_no_steps is not None:
            self.total_no_steps = new_total_no_steps

        # update current_step
        self.current_step = new_current_step if new_current_step is not None else self.current_step + 1

        # calculate progress and expected time
        progress = self.current_step / self.total_no_steps

        elapsed_time = time.time() - self.start_time
        expected_time = elapsed_time / progress * (1 - progress) if progress else 0

        progress_bar_length = 20
        progress_done = int(progress * progress_bar_length)
        progress_bar_string = '=' * progress_done + ' ' * (progress_bar_length - progress_done)

        remaining_time_string = ProgressBar.format_delta_time(expected_time) if expected_time != 0 else "?"

        # print progress bar
        print(f"\x1b[1K\r  {self.task_description}: {self.current_step}/{self.total_no_steps} [{progress_bar_string}] "
              f"{progress * 100 : .0f} %, eta {remaining_time_string}", end="", flush=True)


    def finish(self):
        """
        Finish progress bar.

        Returns:
            None
        """

        # clear line
        print(f"\x1b[1K\r  {self.task_description} [{ProgressBar.format_delta_time(time.time() - self.start_time)}]")
