from torch.utils.data import DataLoader
from torchvision import transforms as T
from Dataset import ImageDataset
import torch.optim as optim
import torch.nn as nn
from torch import save, load
from utilities import instantiate_network, plot_losses
import json
import time
from os import makedirs
from os.path import join
from torchinfo import summary


def train_model(color_dir, perc, folder_name, file_name, architecture,
                gray_dir=None, epochs=50, learning_rate=0.001, model=None):
    """
    Trains the model with the given examples.
    :param color_dir: The directory where the colored images are stored.
    :param perc: The percentage of images, between 0 and 1, to use from `color_dir`.
    :param folder_name: The folder name to be used to store the images.
    :param file_name: The file name prefix to be used for storing model and results in the folder_name.
    :param architecture: The architecture of the model.
    :param gray_dir: The directory where the gray images are stored.
    :param epochs: The number of epochs.
    :param learning_rate: The learning rate.
    :param model: (Optional) The pre-trained model.
    """

    # Loading Dataset and creating the corresponding DataLoader.
    training_data = ImageDataset(color_dir=color_dir, perc=perc, gray_dir=gray_dir)
    train_data_loader = DataLoader(training_data, batch_size=20, shuffle=True)
    print(f"Training on {len(training_data)} images.")

    cnn = instantiate_network(architecture)

    # print(summary(cnn, (32, 1, 600, 800), verbose=1,
    #                          col_names=["input_size", "kernel_size", "output_size", "num_params"]))
    # exit()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    running_losses = []
    all_losses = {}  # Dictionary for epoch -> loss

    # Checking if there is a pre-trained model to be loaded.
    if model is not None:
        checkpoint = load(model)
        cnn.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        cnn.eval()
        cnn.train()
        initial_time = checkpoint["time"]
        running_losses = checkpoint["running_losses"]
    else:
        initial_time = 0

    print(f"Number of parameters: {sum(p.numel() for p in cnn.parameters())}")
    print(f"Running {epochs} epoch(s)")
    print(f"Each epoch runs for {len(train_data_loader)} iterations")
    print()

    start = time.time()
    for epoch in range(epochs):
        epoch_running_loss = 0

        for i, data in enumerate(train_data_loader, 0):
            gray, color = data
            gray = gray.float()
            color = color.float()
            outputs = cnn(gray)

            optimizer.zero_grad()
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item()/ 2000:.3f}')

        running_losses.append(epoch_running_loss)
        all_losses[epoch] = epoch_running_loss

    print("Finished")
    results = {"losses": running_losses}

    # Store losses in json file
    if folder_name is not None:
        figures_dir = join(folder_name, 'figures')
    else:
        figures_dir = 'figures'

    makedirs(figures_dir, exist_ok=True)
    f_name = join(figures_dir, f"training_{file_name}_{learning_rate}_losses.json")

    with open(f_name, "w+") as results_file:
        json.dump(results, results_file)
    print(f"Losses saved at: {f_name}")

    # Store model
    if folder_name is not None:
        model_dir = join(folder_name, 'models')
    else:
        model_dir = 'models'

    makedirs(model_dir, exist_ok=True)
    full_model_path = join(model_dir, f"{file_name}_{learning_rate}_{architecture}_full.pt")

    save({
        "epoch": epoch,
        "model_state_dict": cnn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "time": time.time() - start + initial_time,
        "running_losses": running_losses
    }, full_model_path)
    print(f"Model saved at: {full_model_path}")

    # Plot and save the losses per epoch
    f_name = join(figures_dir, f"loss_per_epoch_training_{file_name}_{learning_rate}_{architecture}.png")
    plot_losses(losses=all_losses,
                title="Losses per epoch",
                path=f_name)
    print(f"A plot of losses per epoch is saved at : {f_name}")

    return full_model_path
