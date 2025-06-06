# -*- coding: utf-8 -*-

"""
Python 3
04 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from quartopy import BotAI, play_games

from collections import defaultdict
from utils.logger import logger
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
        results_p1 = play_games(
            matches=matches // 2,
            player1=player,
            player2=rival,
            verbose=verbose,
            match_dir=match_dir,
            return_file_paths=False,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        )
        logger.debug(results_p1)
        results[idx]["wins"] += results_p1["P1"]
        results[idx]["losses"] += results_p1["P2"]
        results[idx]["draws"] += results_p1["Empates"]

        results_p2 = play_games(
            matches=matches // 2,
            player1=rival,
            player2=player,
            verbose=verbose,
            match_dir=match_dir,
            return_file_paths=False,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        )
        logger.debug(results_p2)
        results[idx]["wins"] += results_p2["P2"]
        results[idx]["losses"] += results_p2["P1"]
        results[idx]["draws"] += results_p2["Empates"]

    return results
