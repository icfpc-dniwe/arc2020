from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

from .task import Task
from typing import Optional


def plot_one(ax: Optional[plt.Axes], task_matrix: np.ndarray, is_input: bool, is_train: bool) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(3, 3))
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
         '#FFFFFF'])
    norm = colors.Normalize(vmin=0, vmax=10)
    ax.imshow(task_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(task_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(task_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    train_title = 'train' if is_train else 'test'
    input_title = 'input' if is_input else 'output'
    ax.set_title(f'{train_title} {input_title}')
    return ax


def plot_task(task: Task) -> None:
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task.train)
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2), squeeze=False)
    for i in range(num_train):
        plot_one(axs[0, i], task.train[i][0], True, True)
        plot_one(axs[1, i], task.train[i][1], False, True)
    plt.tight_layout()
    plt.show()
    num_test = len(task.test)
    fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2), squeeze=False)
    for i in range(num_test):
        plot_one(axs[0, i], task.test[i][0], True, False)
        plot_one(axs[1, i], task.test[i][1], False, False)
    plt.tight_layout()
    plt.show()
