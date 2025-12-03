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

# Handle both relative and absolute imports for better compatibility
try:
    from .NN_abstract import NN_abstract
except ImportError:
    # Fallback for direct execution
    try:
        from NN_abstract import NN_abstract
    except ImportError:
        # Alternative path resolution
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        from NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------- logging --------------------------
try:
    from utils.logger import logger
except ImportError:
    # Fallback for direct execution - create a simple logger
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class QuartoCNN(NN_abstract):
    """
    QuartoCNN is a Convolutional Neural Network (CNN) model for the Quarto board game that predicts the action value function.
    It has a dual-head parallel output architecture:
        * Both outputs use the same backbone features for consistent processing
        * The first output predicts the board position to place a piece
        * The second output predicts the piece to select (independent of board prediction)
    # Input:
    * batchx16x4x4 input tensors representing different positions of the game board.
    * batchx16 dims for each piece

    # Output:
    * batch-16 [-1,1] tensor representing the action value of the board position
    * batch-16 [-1,1] tensor representing the action value of the piece selection
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
        #-----inicio de la modificacion--- FASE 3: Añadir BatchNormalization después de conv1
        # Valor anterior: No existía
        self.bn1 = nn.BatchNorm2d(k1_size)
        #-----fin de la modificacion---

        k2_size = 32
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        #-----inicio de la modificacion--- FASE 3: Añadir BatchNormalization después de conv2
        # Valor anterior: No existía
        self.bn2 = nn.BatchNorm2d(k2_size)
        #-----fin de la modificacion---

        #-----inicio de la modificacion--- FASE 3: Arquitectura más profunda 256→256→128→64
        # Valor anterior: n_neurons = 128, solo fc1
        self.fc1 = nn.Linear(k2_size * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc1b = nn.Linear(256, 256)
        self.bn_fc1b = nn.BatchNorm1d(256)
        self.fc1c = nn.Linear(256, 128)
        self.bn_fc1c = nn.BatchNorm1d(128)
        self.fc1d = nn.Linear(128, 64)
        self.bn_fc1d = nn.BatchNorm1d(64)
        n_neurons_final = 64
        #-----fin de la modificacion---

        # Both heads now receive the same input size for architectural consistency
        # Predicts board position - normalized input (64 dimensions)
        #-----inicio de la modificacion--- FASE 3: Actualizar entrada de cabezas
        # Valor anterior: self.fc2_board = nn.Linear(n_neurons, 4 * 4), n_neurons=128
        self.fc2_board = nn.Linear(n_neurons_final, 4 * 4)

        # Predicts piece selection - normalized input (same 64 dimensions as board)
        # Valor anterior: self.fc2_piece = nn.Linear(n_neurons, 4 * 4), n_neurons=128
        self.fc2_piece = nn.Linear(n_neurons_final, 4 * 4)
        #-----fin de la modificacion---

        #-----inicio de la modificacion--- FASE 3: Reducir Dropout
        # Valor anterior: self.dropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.2)  # Menor dropout para red más profunda
        #-----fin de la modificacion---

    def forward(
        self, x_board: torch.Tensor, x_piece: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        Args:
            ``x_board``: Input tensor of the board with placed pieces (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of selected piece to place (batch_size, 16).
        Returns:
            qav_board: Onehot tensor [-1, 1] of the action value for the board position to place piece (batch_size, 16).
            qav_piece: Onehot tensor [-1, 1] of the action value for the selected piece (batch_size, 16).
        """
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)  # type: ignore

        #-----inicio de la modificacion--- FASE 3: Usar BatchNormalization en convoluciones
        # Valor anterior: x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv1(x)))
        # Valor anterior: x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(self.conv2(x)))
        #-----fin de la modificacion---

        x = x.flatten(start_dim=1)

        #-----inicio de la modificacion--- FASE 3: Usar arquitectura más profunda con BatchNorm
        # Valor anterior: x = F.relu(self.fc1(x))
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc1b(self.fc1b(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc1c(self.fc1c(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc1d(self.fc1d(x)))
        #-----fin de la modificacion---

        x = self.dropout(x)  # Bx64

        # Both outputs now use the same backbone features for consistency
        # This eliminates the architectural inconsistency and allows parallel computation

        # output 1: board position (batch, 16)
        logits_board = self.fc2_board(x)
        qav_board = F.tanh(logits_board)

        # output 2: selected piece (batch, 16) - now uses same input as board
        logits_piece = self.fc2_piece(x)
        qav_piece = F.tanh(logits_piece)

        return qav_board, qav_piece

    def predict(
        self,
        x_board: torch.Tensor,
        x_piece: torch.Tensor,
        TEMPERATURE: float = 1.0,
        DETERMINISTIC: bool = True,
    ):
        """
        Predicts the preferred order of the all the board positions and pieces, with optional ``TEMPERATURE`` for randomness.

        Args:
            ``x_board``: Input tensor of shape (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of shape (batch_size, 16).
            ``TEMPERATURE``: Sampling temperature (>0). Lower values make predictions more deterministic.
            ``DETERMINISTIC``: If True, use argmax instead of sampling.

        Returns:
            * ``board_position``: Predicted idx board position (batch_size, 4, 4).
            * ``predicted_piece``: Sampled idx piece indices (batch_size, 16).
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
            qav_board, qav_piece = self.forward(x_board, x_piece)

            # Use tanh outputs directly for deterministic prediction
            if DETERMINISTIC:
                board_indices = torch.argsort(qav_board, descending=True, dim=1)
                piece_indices = torch.argsort(qav_piece, descending=True, dim=1)
                return board_indices, piece_indices
            else:
                # For stochastic prediction, use softmax over tanh outputs and sample
                board_probs = F.softmax(qav_board / TEMPERATURE, dim=1)
                piece_probs = F.softmax(qav_piece / TEMPERATURE, dim=1)
                board_indices = torch.multinomial(
                    board_probs,
                    board_probs.shape[1],  # all possible combinations
                    replacement=False,
                )
                piece_indices = torch.multinomial(
                    piece_probs,
                    piece_probs.shape[1],  # all possible combinations
                    replacement=False,
                )
                return board_indices, piece_indices
