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
from .NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime
from os import path, makedirs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


class QuartoCNN(NN_abstract):
    """
    QuartoCNN is a Convolutional Neural Network (CNN) model for the Quarto board game.
    # Input:
    * batchx16x4x4 input tensors representing different positions of the game board. 16 dims for each piece, 4x4 grid.

    # Output:
    * batchx4x4 logits tensor representing the board position
    * batch-by-16 logits tensor representing the piece
    """

    @property
    def name(self) -> str:
        return "QuartoCNN1"

    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 16, 4, 4)
        # (batch_size, dims, height, width)
        fc_inpiece_size = 16  # must be multiple of 16

        assert fc_inpiece_size % 16 == 0, "fc_inpiece_size must be a multiple of 16"
        self.fc_in_piece = nn.Linear(
            16, fc_inpiece_size
        )  # Input layer for piece features

        k1_size = 16
        self.conv1 = nn.Conv2d(
            16 + fc_inpiece_size // 16, k1_size, kernel_size=3, padding=1
        )
        k2_size = 32
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        n_neurons = 128
        self.fc1 = nn.Linear(k2_size * 4 * 4, n_neurons)

        # Predicts board position
        self.fc2_board = nn.Linear(n_neurons, 4 * 4)

        # piece: in softmax
        self.fc2_piece = nn.Linear(n_neurons, 4 * 4)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self, x_board: torch.Tensor | np.ndarray, x_piece: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        Args:
            x_board: Input tensor of the board with placed pieces (batch_size, 16, 4, 4).
            x_piece: Input tensor of selected piece to place (batch_size, 16).
        Returns:
            logits_board: Output tensor of the board position to place piece (batch_size, 16).
            logits_piece: Output tensor of selected piece (batch_size, 16).
        """
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)  # type: ignore
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # output 1: board position (batch, 16)
        logits_board = self.fc2_board(x)

        # output 2: selected piece (batch, 16)
        logits_piece = self.fc2_piece(x)
        return logits_board, logits_piece

    def predict(
        self,
        x_board: torch.Tensor | np.ndarray,
        x_piece: torch.Tensor | np.ndarray,
        TEMPERATURE: float = 1.0,
        DETERMINISTIC: bool = True,
    ):
        """
        Predict the board position and piece from the input tensor, with optional ``TEMPERATURE`` for randomness.

        Args:
            ``x_board``: Input tensor of shape (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of shape (batch_size, 16).
            ``TEMPERATURE``: Sampling temperature (>0). Lower values make predictions more deterministic.
            ``DETERMINISTIC``: If True, use argmax instead of sampling.

        Returns:
            ``board_position``: Predicted board position (batch_size, 4, 4).

            ``predicted_piece``: Sampled piece indices (batch_size, 16).
        """
        assert x_board.shape[1:] == (
            16,
            4,
            4,
        ), "Input tensor must have shape (batch_size, 16, 4, 4)"
        assert x_piece.shape[1] == 16, "Input tensor must have shape (batch_size, 16)"
        assert (
            x_board.shape[0] == x_piece.shape[0]
        ), "Input tensors must have the same batch size"

        self.eval()
        with torch.no_grad():
            logits_board_position, logits_piece = self.forward(x_board, x_piece)
            # Apply softmax to get probabilities
            board_position_probs = F.softmax(logits_board_position / TEMPERATURE, dim=1)
            piece_probs = F.softmax(logits_piece / TEMPERATURE, dim=1)

            # Compute the outer product for each batch to get all possible combinations
            # board_position_probs: (batch_size, 16)
            # piece_probs: (batch_size, 16)
            # Output: (batch_size, 16, 16) where [i, j, k] = prob(board=j, piece=k)
            batch_size = logits_board_position.shape[0]
            combo_matrix = torch.einsum("bi,bj->bij", board_position_probs, piece_probs)
            logging.debug(combo_matrix)
            combo_matrix = combo_matrix.view(batch_size, -1)

            if DETERMINISTIC:
                preds = torch.argsort(combo_matrix, dim=1, descending=True)
            else:
                # Sample from the multinomial distribution
                preds = torch.multinomial(
                    combo_matrix,
                    num_samples=combo_matrix.shape[1],  # all possible combinations
                    replacement=False,
                )

            # Convert the flat indices in preds to (board, piece) indices
            # Each index corresponds to (board_idx * 16 + piece_idx)
            board_indices = preds // 16
            piece_indices = preds % 16

            return board_indices, piece_indices


def train(
    model,
    dataloader,
    optimizer,
    criterion_board,
    criterion_piece,
    device="cpu",
    epochs=10,
    log_interval=10,
):
    """
    Train the QuartoCNN model.

    Args:
        model: QuartoCNN instance.
        dataloader: DataLoader yielding (input, board_target, piece_target).
        optimizer: torch optimizer.
        criterion_board: loss function for board position (e.g., nn.CrossEntropyLoss or nn.MSELoss).
        criterion_piece: loss function for piece prediction (e.g., nn.BCELoss).
        device: 'cpu' or 'cuda'.
        epochs: number of epochs.
        log_interval: batches per log.
    """
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (inputs, board_targets, piece_targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            board_targets = board_targets.to(device)
            piece_targets = piece_targets.to(device)

            optimizer.zero_grad()
            board_pred, piece_pred = model(inputs)

            # Flatten board_targets if using CrossEntropyLoss
            if isinstance(criterion_board, nn.CrossEntropyLoss):
                board_pred_flat = board_pred.view(-1, 16)
                board_targets_flat = board_targets.view(-1)
                loss_board = criterion_board(board_pred_flat, board_targets_flat)
            else:
                loss_board = criterion_board(board_pred, board_targets)

            loss_piece = criterion_piece(piece_pred, piece_targets)
            loss = loss_board + loss_piece
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
