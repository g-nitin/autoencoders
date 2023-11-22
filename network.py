import torch.nn as nn
import torch.nn.functional as F
from torch import cat


class Network1(nn.Module):
    def __init__(self):
        """
        Initializes each part of the convolutional neural network.
        """

        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        # self.conv4_bn = nn.BatchNorm2d(256)

        # Dilation layers.
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=3, dilation=2)
        self.conv5_bn = nn.BatchNorm2d(128)

        # Decoder
        # self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.t_conv1_bn = nn.BatchNorm2d(128)
        self.t_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(64)
        self.t_conv2 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm2d(32)
        self.t_conv3 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        # Output layer
        self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

        # self.t_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # self.t_conv1_bn = nn.BatchNorm2d(128)
        # self.t_conv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        # self.t_conv2_bn = nn.BatchNorm2d(64)
        # self.t_conv3 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        # self.t_conv3_bn = nn.BatchNorm2d(32)
        # self.t_conv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)
        #
        # self.output = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x_1 = F.relu(self.conv1_bn(self.conv1(x)))
        x_2 = F.relu(self.conv2_bn(self.conv2(x_1)))
        x_3 = F.relu(self.conv3_bn(self.conv3(x_2)))

        # Dilation layers.
        x_4 = F.relu(self.conv4_bn(self.conv4(x_3)))
        x_4_d = F.relu(self.conv5_bn(self.conv5(x_4)))

        x_5 = F.relu(self.t_conv1_bn(self.t_conv1(x_4_d)))
        x_5 = cat((x_5, x_2), 1)

        x_6 = F.relu(self.t_conv2_bn(self.t_conv2(x_5)))
        x_6 = cat((x_6, x_1), 1)

        x_7 = F.relu(self.t_conv3(x_6))
        x_7 = cat((x_7, x), 1)

        x = self.output(x_7)
        return x

        # # Encoding
        # x = self.conv1_bn(self.conv1(x))
        # x = self.conv2_bn(self.conv2(x))
        # x = self.conv3_bn(self.conv3(x))
        # x = self.conv4_bn(self.conv4(x))
        # x = self.conv5_bn(self.conv5(x))
        #
        # # Decoding
        # x = self.t_conv1_bn(self.t_conv1(x))
        # x = self.t_conv2_bn(self.t_conv2(x))
        #
        # # Output layer
        # x = self.t_conv3(x)
        # return x
