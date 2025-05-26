# -*- coding: utf-8 -*-

"""
CNN_bot - Bot based in CNN to play Quarto
"""

"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from ..models.CNN1 import QuartoCNN

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


from quartopy import logger, BotAI, Piece, QuartoGame


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        return "CNN_bot"

    def __init__(self, model_path: str | None = None):
        """
        Initializes the CNN bot.
        ## Parameters
        ``model_path``: str | None
            Path to the pre-trained model. If None, random weights are loaded.
        """
        super().__init__()  # aunque no hace nada
        logger.debug(f"CNN_bot initialized")

        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = QuartoCNN.from_file(model_path)
        else:
            logger.info("Loading model with random weights")
            self.model = QuartoCNN()
        logger.debug("Model loaded successfully")

    def select(self, game: QuartoGame, _, *args, **kwargs) -> Piece:
        """Selects a piece from the storage."""
        board_matrix = game.game_board.encode()

        valid_moves: list[tuple[int, int]] = game.storage_board.get_valid_moves()  # type: ignore
        valid_pieces = game.storage_board.get_valid_pieces()

        assert valid_moves, "No valid moves available in storage."

        print(*zip(range(len(valid_pieces)), valid_pieces), sep="\n")
        option = input(f"Select a piece by number [0-{len(valid_moves)-1}]: ")
        try:
            option = int(option)
            if option < 0 or option >= len(valid_moves):
                raise ValueError("Invalid option selected.")
        except ValueError as e:
            logger.error(f"Invalid input: {e}. Defaulting to first valid piece.")
            option = 0
        r, c = valid_moves[option]
        selected_piece = game.storage_board.get_piece(r, c)
        logger.debug(f"RandomBot selected piece: {selected_piece} from storage.")
        return selected_piece

    def place_piece(
        self, game: QuartoGame, piece: Piece, ith_option: int = 0, *args, **kwargs
    ) -> tuple[int, int]:
        """Places the selected piece on the game board at a random valid position."""
        valid_moves = game.game_board.get_valid_moves()

        assert valid_moves, "No valid moves available on the game board."

        print(*zip(range(len(valid_moves)), valid_moves), sep="\n")
        option = input(f"Select a coordinate [0-{len(valid_moves)-1}]: ")
        try:
            option = int(option)
            if option < 0 or option >= len(valid_moves):
                raise ValueError("Invalid option selected.")
        except ValueError as e:
            logger.error(f"Invalid input: {e}. Defaulting to first valid piece.")
            option = 0
        position: tuple[int, int] = valid_moves[option]  # type: ignore
        logger.debug(
            f"RandomBot placed piece {piece} at position {position} on the game board."
        )
        return position
