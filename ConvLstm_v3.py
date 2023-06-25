import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import linecache
import csv
import pandas as pd
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_size, padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        self.conv = nn.Conv2d(in_channels=in_channels + out_channels, out_channels=4 * out_channels,
                              kernel_size=kernels_size, padding=padding)
        stdv = 1.0 / math.sqrt(out_channels) if out_channels > 0 else 0
        self.W_ci = nn.Parameter(torch.FloatTensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.FloatTensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.FloatTensor(out_channels, *frame_size))

        nn.init.uniform_(self.W_ci, -stdv, stdv)
        nn.init.uniform_(self.W_co, -stdv, stdv)
        nn.init.uniform_(self.W_cf, -stdv, stdv)




    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = torch.sigmoid(o_conv + self.W_co * C)
        H = output_gate * self.activation(C)
        return H, C


class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, device):
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, activation, frame_size)

        self.device = device

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, height, width, device=self.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)
            output[:, :, time_step] = H

        return output


class Seq2Seq(nn.Module):

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers, device):
        super(Seq2Seq, self).__init__()
        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size,
                device=device)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size,
                    device=device)
            )

            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
            )

            # Add Convolutional Layer to predict output frame
        self.conv = nn.Sequential(
            nn.Conv2d(num_kernels, num_kernels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_kernels, num_kernels // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(num_kernels // 2, num_channels, 3, 1, 1),
            nn.Tanh()
        )
        # self.conv = nn.Conv2d(
        #     in_channels=num_kernels, out_channels=num_channels,
        #     kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # Forward propagation through all the layers
        output = self.sequential(X)
        B, C, T, H, W = output.size()
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = output.view(B * T, C, H, W)

        # Return only the last output frame
        # output = self.conv(output[:, :, -1])
        output = self.conv(output)
        BT, C, H, W = output.size()
        output = output.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return output