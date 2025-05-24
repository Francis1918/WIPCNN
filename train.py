# -*- coding: utf-8 -*-

"""
train - Summary

Usage:
    train.py
    train.py -h|--help
    train.py --version

Options:
    -h,--help               show help.
"""

"""
Python 3
30 / 04 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from os import getenv
from dotenv import load_dotenv

load_dotenv()

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


# ----------------------------- #### --------------------------
from docopt import docopt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual Block with BatchNorm"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class SpatialFeatureExtractor(nn.Module):
    """3D CNN for processing board state"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 128, kernel_size=(2, 2, 4))
        self.res_blocks = nn.Sequential(ResidualBlock3D(128), ResidualBlock3D(128))
        self.attn_pool = nn.Linear(128, 1)  # Attention pooling

    def forward(self, x):
        # Input: (B, 1, 4, 4, 4)
        x = F.relu(self.conv1(x))  # (B, 128, 3, 3, 1)
        x = self.res_blocks(x)

        # Attention pooling
        batch, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        attn_weights = F.softmax(self.attn_pool(x), dim=1)
        return (x * attn_weights).sum(dim=(1, 2, 3))  # (B, 128)


class PieceInteractionNetwork(nn.Module):
    """Transformer-based piece relationship model"""

    def __init__(self):
        super().__init__()
        self.piece_embed = nn.Linear(4, 64)  # Embed individual pieces
        self.selected_embed = nn.Linear(4, 64)

        # Transformer components
        self.self_attn = nn.MultiheadAttention(64, 4)
        self.cross_attn = nn.MultiheadAttention(64, 4)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, selected_piece, available_pieces, spatial_feat):
        # Embed pieces
        available_emb = self.piece_embed(available_pieces)  # (B, 16, 64)
        selected_emb = self.selected_embed(selected_piece).unsqueeze(1)  # (B, 1, 64)

        # Self-attention over pieces
        tokens = torch.cat([selected_emb, available_emb], dim=1).permute(1, 0, 2)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # Cross-attention with spatial features
        spatial_feat = spatial_feat.unsqueeze(0)  # (1, B, 128)
        cross_out, _ = self.cross_attn(tokens, spatial_feat, spatial_feat)
        tokens = self.norm2(tokens + cross_out)

        return tokens.mean(dim=0)  # (B, 64)


class QuartoNet(nn.Module):
    """Complete Quarto Architecture"""

    def __init__(self):
        super().__init__()
        self.spatial_net = SpatialFeatureExtractor()
        self.piece_net = PieceInteractionNetwork()

        # Fusion and decision layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 2, 256),  # +2 for phase indicator
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )

        # Output heads
        self.placement_head = nn.Linear(256, 16)
        self.selection_head = nn.Linear(256, 16)
        self.value_head = nn.Linear(256, 1)

    def forward(self, board, selected_piece, available_pieces, phase):
        # Input processing
        spatial_feat = self.spatial_net(board.unsqueeze(1))  # (B, 128)
        piece_feat = self.piece_net(selected_piece, available_pieces, spatial_feat)

        # Concatenate features + phase
        combined = torch.cat([spatial_feat, piece_feat, phase], dim=1)
        hidden = self.fc(combined)

        # Outputs
        place_logits = self.placement_head(hidden)
        select_logits = self.selection_head(hidden)
        value = torch.tanh(self.value_head(hidden))

        return place_logits, select_logits, value


# ####################################################################
def main(args):
    # Mock input data
    batch_size = 4
    board = torch.randn(batch_size, 4, 4, 4)  # Board state
    selected_piece = torch.randn(batch_size, 4)  # Current piece to place
    available_pieces = torch.randn(batch_size, 16)  # Available pieces
    phase = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]])  # Phase indicator

    model = QuartoNet()
    p_logits, s_logits, value = model(board, selected_piece, available_pieces, phase)

    print("Placement logits shape:", p_logits.shape)
    print("Selection logits shape:", s_logits.shape)
    print("Value shape:", value.shape)


if __name__ == "__main__":
    args = docopt(
        doc=__doc__,
        version="1",
    )
    logging.debug(args)
    main(args)
