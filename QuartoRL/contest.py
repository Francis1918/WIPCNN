# -*- coding: utf-8 -*-

"""
Python 3
04 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# Handle quartopy import with robust fallback system
try:
    from quartopy import BotAI, play_games
except ImportError:
    # Fallback for direct execution or missing quartopy
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
        from quartopy import BotAI, play_games

        print("✅ Quartopy imported successfully after dependency setup")

    except ImportError as e:
        error_msg = (
            "❌ DEPENDENCY ERROR: Cannot import quartopy\n\n"
            "🔧 TROUBLESHOOTING STEPS:\n"
            "1. Ensure the 'quartopy' project is available in your environment\n"
            "2. Check if quartopy is in one of these locations:\n"
            "   - ../quartopy (relative to this project)\n"
            "   - ~/Documents/GitHub/Quartopy\n"
            "   - C:/Users/bravo/Documents/quartopy\n"
            "3. If quartopy is elsewhere, create a .env file with:\n"
            "   QUARTOPY_PATH=/path/to/your/quartopy/project\n"
            "4. Or install quartopy as a package: pip install quartopy\n\n"
            f"📋 Original error: {e}\n\n"
            "💡 For more help, check setup_dependencies.py"
        )
        print(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"❌ UNEXPECTED ERROR during quartopy setup: {e}\n\n"
            "🔧 SUGGESTED ACTIONS:\n"
            "1. Check that setup_dependencies.py exists and is valid\n"
            "2. Verify file permissions in the project directory\n"
            "3. Try running the project with administrator privileges\n"
            "4. Check the utils/logger.py for any issues\n\n"
            "💡 Consider manually adding quartopy to your Python path"
        )
        print(error_msg)
        raise ImportError(error_msg) from e

from collections import defaultdict

# Handle logger import with fallback
try:
    from utils.logger import logger
except ImportError:
    # Fallback for direct execution - create a simple logger
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

import random


# ####################################################################
def run_contest(
    player: BotAI,
    rivals: list[str],
    rival_class: type[BotAI],
    matches: int = 100,
    rivals_clip: int = -1,
    verbose: bool = True,
    match_dir: str = "./partidas_guardadas/",
    PROGRESS_MESSAGE: str = "Playing tournament matches...",
):
    """Run a contest between a player and multiple rivals.
    Args:
        player (BotAI): The player bot.
        rivals (list[str]): List of file paths to rival bots.
        rival_class (type[BotAI]): Class type of the rival bots.
        matches (int): Total number of matches to play against each rival.
        rivals_clip (int): Limit the number of rivals to consider. -1 means no limit.
        verbose (bool): Whether to print detailed logs.
        match_dir (str): Directory to save match files.
    """
    n_rivals = len(rivals)
    logger.debug(f"Running contest with {n_rivals} rivals, {matches} matches")

    selected = range(n_rivals)  # Default to all rivals
    if rivals_clip == -1:
        logger.debug("No clipping of rivals, using all available rivals")
    elif rivals_clip > n_rivals:
        logger.debug(
            f"Cannot clip to requested {rivals_clip}. Playing against all {n_rivals} rivals"
        )

    else:
        logger.debug(f"Clipping rivals to {rivals_clip} random rivals")
        selected = sorted(random.sample(range(n_rivals), k=rivals_clip))

    rivals_selected = {i: rivals[i] for i in selected}

    # index del rival: {"wins": 0, "losses": 0, "draws": 0}
    results: dict[int, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "draws": 0}
    )
    for idx, rival_file in rivals_selected.items():
        rival = rival_class(model_path=rival_file)

        logger.debug(f"Playing against rival {idx + 1}/{len(rivals)}: {rival.name}")

        # Create unique subdirectories to avoid race conditions and file conflicts
        # When player is P1 (playing first)
        match_dir_p1 = f"{match_dir.rstrip('/')}/rival_{idx:03d}_player_as_p1/"
        results_p1 = play_games(
            matches=matches // 2,
            player1=player,
            player2=rival,
            verbose=verbose,
            match_dir=match_dir_p1,
            return_file_paths=False,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        )
        logger.debug(f"Results P1 vs rival {idx}: {results_p1}")
        results[idx]["wins"] += results_p1["P1"]
        results[idx]["losses"] += results_p1["P2"]
        results[idx]["draws"] += results_p1["Empates"]

        # When player is P2 (playing second)
        match_dir_p2 = f"{match_dir.rstrip('/')}/rival_{idx:03d}_player_as_p2/"
        results_p2 = play_games(
            matches=matches // 2,
            player1=rival,
            player2=player,
            verbose=verbose,
            match_dir=match_dir_p2,
            return_file_paths=False,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        )
        logger.debug(f"Results P2 vs rival {idx}: {results_p2}")
        results[idx]["wins"] += results_p2["P2"]
        results[idx]["losses"] += results_p2["P1"]
        results[idx]["draws"] += results_p2["Empates"]

    return results
