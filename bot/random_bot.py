# -*- coding: utf-8 -*-


"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from utils.logger import logger as utils_logger  # Rename to avoid conflicts

def _validate_and_import_quartopy():
    """
    Validates and imports quartopy dependencies with clear error messages.

    Returns:
        tuple: (BotAI, Piece, QuartoGame) classes from quartopy

    Raises:
        ImportError: If quartopy cannot be imported with helpful instructions
    """
    try:
        from quartopy import BotAI, Piece, QuartoGame
        utils_logger.debug("✅ Quartopy imported successfully")
        return BotAI, Piece, QuartoGame

    except ImportError as initial_error:
        utils_logger.warning("⚠️ Initial quartopy import failed, attempting dependency setup...")

        # Attempt fallback with setup_dependencies
        try:
            import sys
            from pathlib import Path

            # Add parent directory to path for setup_dependencies
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            # Import and run dependency setup
            import setup_dependencies
            setup_dependencies.setup_quartopy(silent=False)

            # Retry import after setup
            from quartopy import BotAI, Piece, QuartoGame
            utils_logger.info("✅ Quartopy imported successfully after dependency setup")
            return BotAI, Piece, QuartoGame

        except ImportError as final_error:
            error_msg = (
                "❌ DEPENDENCY ERROR: Cannot import quartopy. "
                "Please ensure quartopy is properly installed."
            )
            utils_logger.error(error_msg)
            raise ImportError(error_msg) from final_error

# Import quartopy components
BotAI, Piece, QuartoGame = _validate_and_import_quartopy()

from random import choice


class Quarto_random_bot(BotAI):
    @property
    def name(self) -> str:
        return "random_bot"

    def __init__(self):
        utils_logger.debug(f"RandomBot initialized with name: {self.name}")

    def select(self, game: QuartoGame, ith_option: int = 0, *args, **kwargs) -> Piece:
        """Selects a random piece from the storage."""
        valid_moves = game.storage_board.get_valid_moves()

        assert valid_moves, "No valid moves available in storage."

        r, c = choice(valid_moves)
        selected_piece = game.storage_board.get_piece(r, c)
        utils_logger.debug(f"RandomBot selected piece: {selected_piece} from storage.")
        return selected_piece

    def place_piece(
        self, game: QuartoGame, piece: Piece, ith_option: int = 0, *args, **kwargs
    ) -> tuple[int, int]:
        """Places the selected piece on the game board at a random valid position."""
        valid_moves = game.game_board.get_valid_moves()

        assert valid_moves, "No valid moves available on the game board."

        position = choice(valid_moves)
        utils_logger.debug(
            f"RandomBot placed piece {piece} at position {position} on the game board."
        )
        return position
