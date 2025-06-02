# -*- coding: utf-8 -*-

"""
Python 3
01 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from utils.logger import logger
from datetime import datetime

from quartopy import play_games, BotAI


# ####################################################################
def create_SyncDataCollector(
    env,
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    experiment_name: str,
    number_of_matches: int = 1000,
    steps_per_batch: int = 1600,
):
    """Creates a SyncDataCollector imitator instance for the QuartoRL board game."""
    logger.info(
        "Creating SyncDataCollector imitator instance for QuartoRL board game..."
    )

    batch_size = steps_per_batch

    match_dir = f"./partidas_guardadas/{experiment_name}/{datetime.now().strftime('%Y%m%d_%H%M')}/"

    results = play_games(
        matches=number_of_matches,
        player1=p1_bot,
        player2=p2_bot,
        delay=0,
        verbose=True,
        match_dir=match_dir,
    )

    logger.info(
        f"SyncDataCollector imitator instance created successfully. Matches played: {number_of_matches}, Steps per batch: {steps_per_batch}"
    )

    action: ...
    done: ...
    observation: ...
    rewatd: ...

    return
