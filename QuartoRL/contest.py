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


# ####################################################################
def run_contest(
    player: BotAI,
    rivals: list[str],
    rival_class: type[BotAI],
    matches: int = 100,
    verbose: bool = True,
    match_dir: str = "./partidas_guardadas/",
):
    logger.debug(f"Running contest with {len(rivals)} rivals, {matches} matches")

    # index del rival: {"wins": 0, "losses": 0, "draws": 0}
    results: dict[int, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "draws": 0}
    )
    for idx, rival_file in enumerate(rivals):
        rival = rival_class(model_path=rival_file)

        logger.debug(f"Playing against rival {idx + 1}/{len(rivals)}: {rival.name}")
        results_p1 = play_games(
            matches=matches // 2,
            player1=player,
            player2=rival,
            verbose=verbose,
            match_dir=match_dir,
            return_file_paths=False,
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
        )
        logger.debug(results_p2)
        results[idx]["wins"] += results_p2["P2"]
        results[idx]["losses"] += results_p2["P1"]
        results[idx]["draws"] += results_p2["Empates"]

    return results
