import torch.nn as nn
# from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import load, no_grad, cat
from torch import from_numpy
from dataset import ImageDataset
from utilities import instantiate_network, register_hooks
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from os import makedirs
from os.path import join
import json


def test_model(model_file, color_dir, perc, folder_name, file_name, architecture, gray_dir=None, show_images=True):
    """
    Tests the given model against the given images.
    :param model_file: The file where the model is stored.
    :param color_dir: The directory where the colored images are stored.
    :param perc: The percentage of images, between 0 and 1, to use from `color_dir`.
    :param folder_name: The folder name to be used to store the images.
    :param file_name: The file name prefix to be used to save the images in the folder_name.
    :param architecture: The architecture of the model.
    :param gray_dir: The directory where the gray images are stored.
    :param show_images: Determines if the comparison images will be saved.
    """

    # Loading Dataset and creating the corresponding DataLoader.
    data = ImageDataset(color_dir=color_dir, perc=perc, gray_dir=gray_dir)
    data_loader = DataLoader(data, batch_size=20, shuffle=False, num_workers=8)
    print(f"Testing on {len(data)} images.")

    # Loading the NN and passing the data.
    cnn = instantiate_network(architecture)
    cnn.load_state_dict(load(model_file)["model_state_dict"])

    # Grid setup.
    columns = 4
    rows = 3

    data_iterator = iter(data_loader)

    total_loss = 0

    # Creating an image grid to showcase the results.
    j = 0
    while True:
        gray, color = next(data_iterator, None) or (None, None)
        if gray is None:
            break

        outputs = cnn(gray.float())
        outputs = outputs.int()

        # Removing the channel.
        gray_images = gray.squeeze()

        if show_images:

            fig = plt.figure(figsize=(8, 8))

        # First row consists of the gray images.
        for i in range(columns):
            if show_images:
                fig.add_subplot(rows, columns, i + 1)
            img = gray_images[i]

            # Setting the title on top of the first image.
            if i == 0:
                plt.title("Input")

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            if show_images:
                plt.imshow(img, cmap="gray")

        # Second row consists of the actual colored images.
        for i in range(columns):
            if show_images:
                fig.add_subplot(rows, columns, (i + 1) + columns)

            # Setting the title on top of the first image.
            if i == 0:
                plt.title("Actual")

            if gray_dir is not None:
                img = color[i].permute(1, 2, 0)
            else:
                img = cat((gray[i], color[i]), 0)
                img = lab2rgb(img.permute(1, 2, 0))

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            if show_images:
                plt.imshow(img)

        # Third row consists of the predicted colored images.
        with no_grad():
            for i in range(columns):
                if show_images:
                    fig.add_subplot(rows, columns, (i + 1) + 2 * columns)

                mse = nn.MSELoss()

                if gray_dir is not None:
                    img = outputs[i].permute(1, 2, 0)
                    loss = mse(outputs[i], color[i])
                else:
                    img = cat((gray[i], outputs[i]), 0)
                    img = lab2rgb(img.permute(1, 2, 0))
                    img_true = cat((gray[i], color[i]), 0)
                    img_true = lab2rgb(img_true.permute(1, 2, 0))
                    loss = mse(from_numpy(img).permute(2, 0, 1), from_numpy(img_true).permute(2, 0, 1))

                total_loss += loss.item()

                plt.gca().get_yaxis().set_visible(False)
                plt.gca().set_xticks([])
                plt.gca().set_xlabel(f"Loss: {loss.item():.5f}")

                # Setting the title on top of the first image.
                if i == 0:
                    plt.title("Predicted")

                if show_images:
                    plt.imshow(img)

        output_dir = ""
        if folder_name is not None and show_images:
            makedirs(folder_name, exist_ok=True)
            output_dir = join(folder_name, 'output')

        elif show_images:
            output_dir = 'output'

        makedirs(output_dir, exist_ok=True)
        f_name = join(output_dir, f"testing_{file_name}_{j}_{architecture}.png")

        plt.savefig(f_name)
        print(f"Saving the input, actual, and predicted images at: {f_name}")

        plt.close(plt.gcf())
        j += 1

    # Store total loss in json file
    if folder_name is not None:
        figures_dir = join(folder_name, 'figures')
    else:
        figures_dir = 'figures'

    makedirs(figures_dir, exist_ok=True)
    f_name = join(figures_dir, f'testing_{file_name}_total_loss.json')

    with open(f_name, "w") as results_file:
        json.dump({"total_loss": total_loss}, results_file)
    print(f"Total losses saved at: {f_name}")


def calculate_loss(model_file, file_name, folder_name, color_dir, gray_dir=None, architecture=1):
    """
    Calculates the loss of the given model.
    :param model_file: The model file.
    :param file_name: The file name prefix to be used to save images.
    :param folder_name: The folder name to be used to store the images.
    :param color_dir: The directory where the colored images are stored.
    :param gray_dir: The directory where the gray images are stored.
    :param architecture: The architecture of the model.
    """
    # Loading Dataset and creating the corresponding DataLoader.
    data = ImageDataset(color_dir=color_dir, gray_dir=gray_dir)
    data_loader = DataLoader(data, batch_size=20, shuffle=False)

    # Loading the NN and passing the data.
    cnn = instantiate_network(architecture)
    cnn.load_state_dict(load(model_file)["model_state_dict"])

    data_iterator = iter(data_loader)

    total_loss = 0

    # Creating an image grid to showcase the results.
    while True:
        gray, color = next(data_iterator, None) or (None, None)
        if gray is None:
            break

        outputs = cnn(gray.float())
        outputs = outputs.int()

        # Third row consists of the predicted colored images.
        with no_grad():
            for i in range(gray.shape[0]):
                mse = nn.MSELoss()

                if gray_dir is not None:
                    img = outputs[i].permute(1, 2, 0)
                    loss = mse(outputs[i], color[i])
                else:
                    img = cat((gray[i], outputs[i]), 0)
                    img = lab2rgb(img.permute(1, 2, 0))
                    img_true = cat((gray[i], color[i]), 0)
                    img_true = lab2rgb(img_true.permute(1, 2, 0))
                    loss = mse(from_numpy(img).permute(2, 0, 1), from_numpy(img_true).permute(2, 0, 1))
                total_loss += loss.item()

    # Store total loss in json file
    if folder_name is not None:
        figures_dir = join(folder_name, 'figures')
    else:
        figures_dir = 'figures'

    makedirs(figures_dir, exist_ok=True)
    f_name = join(figures_dir, f'testing_{file_name}_total_loss.json')

    with open(f_name, "w") as results_file:
        json.dump({"total_loss": total_loss}, results_file)
    print(f"Total losses saved at: {f_name}")


def render_layer_output(model_file, color_dir, folder_name, file_name, architecture, number_of_filters):
    """
    Renders the output of each layer.
    :param model_file: The file where the model is stored.
    :param color_dir: The directory where the colored images are stored.
    :param folder_name: The folder name prefix to be used to store the images.
    :param file_name: The file name prefix to be used to store the images in the folder_name.
    :param architecture: The architecture of the model.
    :param number_of_filters: The number of filters to be rendered on the images.
    """

    # Loading Dataset and creating the corresponding DataLoader.
    data = ImageDataset(color_dir=color_dir, gray_dir=None)
    data_loader = DataLoader(data, batch_size=20, shuffle=True)

    # Setting up activation hooks.
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Loading the NN and passing the data.
    cnn = instantiate_network(architecture)
    cnn, layers = register_hooks(cnn, architecture, get_activation)
    cnn.load_state_dict(load(model_file)["model_state_dict"])

    gray, color = next(iter(data_loader), None)

    # Feeding the data to the neural network.
    cnn(gray.float())

    # Grid setup.
    columns = 5
    rows = int((number_of_filters + 1)/5 + 1)

    for layer in layers:
        fig = plt.figure(figsize=(8, 8))
        plt.title(layer["name"])
        # Adding the actual colored image.
        with no_grad():
            fig.add_subplot(rows, columns, 1)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            img = cat((gray[0], color[0]), 0)
            img = lab2rgb(img.permute(1, 2, 0))
            plt.imshow(img)

        # First row consists of the gray images.
        for i in range(1, min(number_of_filters, layer["channels"]) + 1):
            fig.add_subplot(rows, columns, i + 1)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.imshow(activation[layer["name"]][0][i - 1])

        if folder_name is not None:
            dir_name = join(folder_name, 'output')
        else:
            dir_name = 'output'

        makedirs(dir_name, exist_ok=True)
        f_name = join(dir_name, f"layers_{file_name}_{layer['name']}.png")
        plt.savefig(f_name)
        print(f"Plot saved at: {f_name}")