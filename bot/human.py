# -*- coding: utf-8 -*-


"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from utils.logger import logger as utils_logger

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
        utils_logger.debug("✅ Quartopy importado correctamente")
        return BotAI, Piece, QuartoGame

    except ImportError as initial_error:
        utils_logger.warning("⚠️ Error al importar quartopy, intentando configurar dependencias...")

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
            utils_logger.info("✅ Quartopy importado correctamente después de configurar dependencias")
            return BotAI, Piece, QuartoGame

        except ImportError as final_error:
            error_msg = (
                "❌ ERROR DE DEPENDENCIA: No se puede importar quartopy. "
                "Asegúrese de que quartopy esté correctamente instalado."
            )
            utils_logger.error(error_msg)
            raise ImportError(error_msg) from final_error

# Import quartopy components
BotAI, Piece, QuartoGame = _validate_and_import_quartopy()


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        return "Human_bot"

    def __init__(self):
        utils_logger.debug(f"Humanbot initialized with name: {self.name}")

    def select(self, game: QuartoGame, ith_option: int = 0, *args, **kwargs) -> Piece:
        """Selects a random piece from the storage."""
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
            utils_logger.error(f"Invalid input: {e}. Defaulting to first valid piece.")
            option = 0
        r, c = valid_moves[option]
        selected_piece = game.storage_board.get_piece(r, c)
        utils_logger.debug(f"RandomBot selected piece: {selected_piece} from storage.")
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
            utils_logger.error(f"Invalid input: {e}. Defaulting to first valid piece.")
            option = 0
        position: tuple[int, int] = valid_moves[option]  # type: ignore
        utils_logger.debug(
            f"RandomBot placed piece {piece} at position {position} on the game board."
        )
        return position
