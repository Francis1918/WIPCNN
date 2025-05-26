# -*- coding: utf-8 -*-


"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""


from quartopy import logger, BotAI, Piece, QuartoGame

from random import choice


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        return "random_bot"

    def __init__(self):
        logger.debug(f"RandomBot initialized with name: {self.name}")

    def select(self, game: QuartoGame, ith_option: int = 0, *args, **kwargs) -> Piece:
        """Selects a random piece from the storage."""
        valid_moves = game.storage_board.get_valid_moves()

        assert valid_moves, "No valid moves available in storage."

        r, c = choice(valid_moves)
        selected_piece = game.storage_board.get_piece(r, c)
        logger.debug(f"RandomBot selected piece: {selected_piece} from storage.")
        return selected_piece

    def place_piece(
        self, game: QuartoGame, piece: Piece, ith_option: int = 0, *args, **kwargs
    ) -> tuple[int, int]:
        """Places the selected piece on the game board at a random valid position."""
        valid_moves = game.game_board.get_valid_moves()

        assert valid_moves, "No valid moves available on the game board."

        position = choice(valid_moves)
        logger.debug(
            f"RandomBot placed piece {piece} at position {position} on the game board."
        )
        return position
