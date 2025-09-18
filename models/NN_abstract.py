# -*- coding: utf-8 -*-
"""Abstract base class for neural network models in the Quarto game.
This class defines the interface for neural network models used in the Quarto game.
It defines the interfaces for model initialization, forward pass and prediction, and it implements model export and import.
"""

"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from abc import ABC, abstractmethod

import torch

from datetime import datetime
from os import path, makedirs
from pathlib import Path


class NN_abstract(ABC, torch.nn.Module):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x_board: torch.Tensor, x_piece: torch.Tensor, *args, **kwargs):
        pass

    @abstractmethod
    def predict(
        self,
        x_board: torch.Tensor,
        x_piece: torch.Tensor,
        TEMPERATURE: float = 1.0,
        DETERMINISTIC: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the board position and piece from the input tensor, with optional ``TEMPERATURE`` for randomness.

        Args:
            ``x_board``: Input tensor of shape (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of shape (batch_size, 16).
            ``TEMPERATURE``: Sampling temperature (>0). Lower values make predictions more deterministic.
            ``DETERMINISTIC``: If True, use argmax instead of sampling.

        Returns:
            board_position: Predicted board position (batch_size, 4, 4).
            predicted_piece: Sampled piece indices (batch_size, 16).
        """
        pass

    # ####################################################################
    @classmethod
    def from_file(cls, weights_path: str):
        """
        Load the model from a file.

        Returns:
            QuartoCNN instance with loaded weights.
        """
        model = cls()

        model.load_state_dict(torch.load(weights_path))

        return model

    def export_model(
        self,
        checkpoint_suffix: str,
        checkpoint_folder: str = "$__filedir__$/weights/",
    ) -> str:
        """
        Export the model to a file with the datetime and suffix in the filename.

        ## Args:
            checkpoint_suffix: Suffix for the checkpoint file name. Usually the epoch number.
            checkpoint_folder: Folder to save the model weights.
            Defaults to "$__filedir__$/weights/", which will be replaced with the directory of this file.
        ## Returns:
            The full path to the saved model file.
        """
        checkpoint_name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M')}-{checkpoint_suffix}.pt"
        )

        if checkpoint_folder.startswith("$__filedir__$/"):
            # Replace the placeholder with the actual directory of this file
            # Replace the placeholder with the actual directory of this file using pathlib for cross-platform compatibility
            base_dir = Path(__file__).parent
            relative_part = checkpoint_folder.replace("$__filedir__$/", "")
            # Remove leading/trailing slashes and convert to Path
            relative_part = relative_part.strip("/\\")
            checkpoint_folder = str(base_dir / relative_part) if relative_part else str(base_dir)
        file_path = path.join(checkpoint_folder, self.name, checkpoint_name)
        # Use pathlib for robust cross-platform path handling
        checkpoint_path = Path(checkpoint_folder)
        file_path = checkpoint_path / self.name / checkpoint_name
        makedirs(path.dirname(file_path), exist_ok=True)
        # Create directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        torch.save(self.state_dict(), str(file_path))

        return str(file_path)
