# IMPORTS #####
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ROUTINES ######
def to_binary_repr_list(num: int, bits: int):
    return [num // (2 ** j) % 2 for j in range(bits)][::-1]


def einsum_subscripts(*initial_subs: str, to=""):
    # function for better code readability:
    # converts the strings of initial subscripts in *args and the strings of final subscripts in **kwargs
    # (these strings can contain backslashes, numbers, primes, etc. like in SB21 and in my notes)
    # (e.g., one arg in *args might be arg="\vq',\lambda',\nu'" or arg="\vq_2',\nu_2'")
    # into one large einstein sum subscripts string for np.einsum
    list_letters = list(map(chr, range(97, 123)))  # list of all letters in the alphabet
    dict_subs = {}  # dictionary which contains to correspondences between subscripts in SB21/my notes
                         # and subscripts conforming with the conventions/requirements of np.einsum
    lists_initial_subs = [subs.split(",") for subs in initial_subs]
    list_final_subs = to.split(",")
    translated_subs = ""

    for subs_list in lists_initial_subs:
        for sub in subs_list:
            if sub not in dict_subs.keys():
                l = len(dict_subs)
                dict_subs.update({sub: list_letters[l]})
            translated_subs += dict_subs[sub]
        translated_subs += ","

    translated_subs = translated_subs[:-1]  # removes the last comma, which is too much
    translated_subs += "->"

    for sub in list_final_subs:
        translated_subs += dict_subs[sub]

    return translated_subs


def import_policy_from_csv(T: int,
                           path='/Users/davidreiss/Desktop/Archiv/Quantum_PG_1/Plots/8_Final_policy_probabilities.csv'):
    # path to CSV file
    csv_file = path

    # create empty list to store data
    data_list_dict = []

    # open and read CSV file
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data_list_dict.append(row)

    # 'data' contains CSV data as a list of dictionaries
    # create empty list to store data
    data_array = np.empty((T, 2 * T - 1))
    data_array[:] = np.nan

    # convert to data structure used in class OptimalPolicyCalculations
    for row in data_list_dict:
        t = int(row["t"])
        x = int(row["x"])
        prob = float(row["probability of action_1"])

        data_array[t, x + T - 1] = prob

    return data_array


def format_delta_time(time_in_seconds: float) -> str:
    seconds: float = time_in_seconds % 60
    minutes: int = int((time_in_seconds // 60) % 60)
    hours: int = int((time_in_seconds // 60 // 24))

    if hours:
        return f"{hours} h {minutes} min"
    elif minutes:
        return f"{minutes} min {seconds:0.1f} s"
    else:
        return f"{seconds:0.1f} s"


def plot_prob_distribution(T: int, prob_array: np.ndarray, set_title=True, title="", plot_mask: np.ndarray = None,
                           diff=False):
    """
    Plots probability distributions as functions of t and x in terms of heat maps
    :param title:
    :param T:
    :param prob_array:
    :return:
    """
    # prepare prob_array for imshow
    if plot_mask is not None:
        prob_array = np.where(plot_mask, np.nan, prob_array)

    prob_array = np.swapaxes(prob_array, 0, 1)
    # now indices x,t
    # prob_array = prob_array[::-1, :]

    if not diff:
        prob_array = 1 - prob_array

    # plot as heat map
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(prob_array, cmap='viridis')

    fig.colorbar(im, cax=cax, orientation='vertical')

    # set title, labels, ticks, and limits
    if set_title:
        ax.set_title(title + "(to go 1 step down)")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")

    ax.set_xticks(np.array([0, 1 * T//4, 2 * T//4, 3 * T//4, T]),
                  labels=[str(0), str(1 * T//4), str(2 * T//4), str(3 * T//4), str(T)])

    ax.set_ylim([1/2 * T - 2, 3/2 * T])
    ax.set_yticks(np.array([2 * T//4 - 1, 3 * T//4 - 1, T - 1, 5 * T//4 - 1, 6 * T//4 - 1]),
                  labels=[str(-2 * T//4), str(-1 * T//4), str(0), str(1 * T//4), str(2 * T//4)])

    # save plot
    fig.savefig(title + "_to_go_1_step_down.pdf", bbox_inches="tight")

    plt.show()

    """
    # plot as weighted graph
    # to this end use
    # https://stackoverflow.com/questions/28372127/add-edge-weights-to-plot-output-in-networkx
    # https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_numpy_array.html

    # construct adjacency matrix
    no_t, no_x = np.shape(P_W_array)
    no_vertices = (no_t + 1) * (no_x + 2)
    adjacency_matrix = np.zeros((no_vertices, no_vertices))

    for t in range(no_t):
        for x in range(no_x):
            vertex = t * no_x + x
            vertex_step_up = (t + 1) * no_x + (x + 1)
            vertex_step_down = (t + 1) * no_x + (x - 1)

            adjacency_matrix[vertex, vertex_step_up] = adjacency_matrix[vertex_step_up, vertex] = \
                P_W_array[t, x]
            adjacency_matrix[vertex, vertex_step_down] = adjacency_matrix[vertex_step_down, vertex] = \
                (1 - P_W_array[t, x])

            graph = nx.from_numpy_array(adjacency_matrix)
            nx.draw(graph)
            plt.show()
    """


class ProgressBar:
    def __init__(self, total: int | None = None, task_description: str = ""):
        self.start_time = time.time()
        self.i = -1
        self.total = total
        self.task_description = task_description

    def update(self, step: int | None = None, total: int | None = None):
        # update total
        if total is not None:
            self.total = total
        self.i = step if step is not None else self.i + 1

        progress = self.i / self.total
        elapsed_time = time.time() - self.start_time
        expected_time = elapsed_time / progress * (1 - progress) if progress else 0

        progress_bar_length = 20
        progress_done = int(progress * progress_bar_length)
        progress_bar_string = '=' * progress_done + ' ' * (progress_bar_length - progress_done)

        remaining_time_string = format_delta_time(expected_time) if expected_time != 0 else "?"

        print(f"\x1b[1K\r  {self.task_description}: {self.i}/{self.total} [{progress_bar_string}] "
              f"{progress * 100 : .0f} %, eta {remaining_time_string}", end="", flush=True)

    def finish(self):
        # clear line
        print(f"\x1b[1K\r  {self.task_description} [{format_delta_time(time.time() - self.start_time)}]")
