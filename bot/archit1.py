# -*- coding: utf-8 -*-

"""
CNN_toy - Is the basic CNN model for the Quarto board game.
"""

"""
Python 3
15 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


input_shape = (1, 4, 4, 4)  # (batch_size, width, height, dimension)

output1_shape = (4, 4)  # board position (width, height)
output2_shape = (4,)  # piece (Color, Forma, Hueco, Tamaño)


import torch.nn as nn
import torch.nn.functional as F


class QuartoCNN(nn.Module):
    def __init__(self):
        super(QuartoCNN, self).__init__()
        # Input shape: (batch_size, 4, 4, 4)
        # PyTorch expects (batch_size, channels, height, width)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.board_position = nn.Linear(128, 16)  # Predicts board position (4x4)
        self.L_piece = nn.Linear(128, 4)  # piece: 4D (Color, Forma, Hueco, Tamaño)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))

        # output 1: board position (4x4)
        board_position = self.board_position(x)  # (batch_size, 16)
        board_position = F.softmax(board_position, dim=1)
        board_position = board_position.view(-1, 4, 4)  # (batch_size, 4, 4)

        # output 2: piece (Color, Forma, Hueco, Tamaño)
        L_piece = self.L_piece(x)  # (batch_size, 4)
        L_piece = F.sigmoid(L_piece)
        return board_position, L_piece
