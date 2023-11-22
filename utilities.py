import matplotlib.pyplot as plt
from network import *


def plot_loss(loss, title, filename):
    """
    Plots the loss for each epoch.
    :param loss: A list with the loss per epoch.
    :param title: The plot title.
    :param filename: The filename.
    """
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.savefig(f"figures/{filename}.png")


def plot_losses2(losses, path):
    """
    Plots multiple losses for each epoch.
    Reference: towardsdatascience.com/5-steps-to-build-beautiful-line-charts-with-python-655ac5477310

    :param losses: A Dictionary with the losses per epoch.
    :param path: The path.
    """

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=96)
    ax.plot(list(losses.keys()), list(losses.values()), zorder=2)

    # Add an end point marker to the last value
    ax.plot(list(losses.keys())[-1], list(losses.values())[-1], 'o', markersize=10, alpha=0.3)
    ax.plot(list(losses.keys())[-1], list(losses.values())[-1], 'o', markersize=5)

    # Create the grid
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('Epoch', fontsize=12, labelpad=10)  # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)

    # Reformat y-axis
    ax.set_ylabel('Losses', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Remove the spines
    ax.spines[['top', 'right']].set_visible(False)

    # Make the left and bottom spine thicker
    ax.spines['bottom'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)

    # Add in red line and rectangle on top
    ax.plot([0.05, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='#FFD700', linewidth=.6)
    ax.add_patch(plt.Rectangle((0.05, .98), 0.04, -0.02, facecolor='#FFD700',
                               transform=fig.transFigure, clip_on=False, linewidth=0))

    # Add in title and subtitle
    ax.text(x=0.05, y=.93, s="Model Loss Per Epoch", transform=fig.transFigure,
            ha='left', fontsize=14, weight='bold', alpha=.8)

    # Adjust the margins around the plot area
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)

    # Set a white background
    fig.patch.set_facecolor('white')

    plt.savefig(path)


def instantiate_network(architecture):
    """
    Instantiates the network that corresponds to the given
    architecture.
    :param architecture: The network architecture.
    :return: The network.
    """
    if architecture == 1:
        return Network1()
    else:
        return Network1()


def register_hooks(model, architecture, hook):
    """
    Registers the given hook to the given model.
    :param model: The model.
    :param architecture: Model's architecture.
    :param hook: The hook to be registered.
    :return: The model instance and the layers list.
    """
    layers = []
    if architecture == 1:
        model.conv1.register_forward_hook(hook("conv1"))
        model.conv2.register_forward_hook(hook("conv2"))
        model.conv3.register_forward_hook(hook("conv3"))
        model.t_conv1.register_forward_hook(hook("t_conv1"))
        model.t_conv2.register_forward_hook(hook("t_conv2"))
        model.t_conv3.register_forward_hook(hook("t_conv3"))
        model.output.register_forward_hook(hook("output"))

        layers = [
            {
                "name": "conv1",
                "channels": model.conv1.out_channels
            },
            {
                "name": "conv2",
                "channels": model.conv2.out_channels
            },
            {
                "name": "conv3",
                "channels": model.conv3.out_channels
            },
            {
                "name": "t_conv1",
                "channels": model.t_conv1.out_channels
            },
            {
                "name": "t_conv2",
                "channels": model.t_conv2.out_channels
            },
            {
                "name": "t_conv3",
                "channels": model.t_conv3.out_channels
            },
            {
                "name": "output",
                "channels": model.output.out_channels
            }
        ]
    return model, layers
