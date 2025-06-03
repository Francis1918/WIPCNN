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

from tensordict import set_list_to_stack, TensorDict

from quartopy import play_games, BotAI, Board

import numpy as np
import pandas as pd


# ####################################################################
def process_match(match_path: str, result: int):
    """Reads a match file and extracts: observation, action, and reward for both players.
    ## Args:
        * match_path (str): Path to the match file.
        * result (int): Result of the match, where 1 is a win for player 1, -1 is a win for player 2, and 0 is a draw.
    ## Returns:
        * p1 (pd.DataFrame): DataFrame containing player 1's observations, actions, and rewards.
        * p2 (pd.DataFrame): DataFrame containing player 2's observations, actions, and rewards.
    """
    df = pd.read_csv(match_path)
    p1_selected = df["Pieza Index"][::4]
    p2_selected = df["Pieza Index"][2::4]
    p1_position = df["Posición Index"][3::4]
    p2_position = df["Posición Index"][1::4]
    # p1_tries = df["Intento"][::4]
    # p2_tries = df["Intento"][2::4]

    assert (len(p1_selected) == len(p2_position)) and (
        len(p1_position) == len(p2_selected)
    ), f"Mismatch in selected pieces and positions length: {len(p1_selected)} vs {len(p1_position)} and {len(p2_selected)} vs {len(p2_position)}"

    p1 = pd.DataFrame()
    p2 = pd.DataFrame()

    p1_obs = pd.concat([pd.Series(["0"]), df["Tablero"][3::4]], ignore_index=True)
    # p1["observation"] = p1_obs.apply(Board.deserialize)  # type: ignore
    # p2["observation"] = df["Tablero"][1::4].apply(Board.deserialize)  # type: ignore
    p1["observation"] = p1_obs
    p2["observation"] = df["Tablero"][1::4].reset_index(drop=True)

    p1_position = pd.concat([pd.Series([-1]), p1_position], ignore_index=True)

    if len(p2_position) == len(p2_selected):
        p1_selected = pd.concat([p1_selected, pd.Series([-1])], ignore_index=True)
    else:
        p2_selected = pd.concat([p2_selected, pd.Series([-1])], ignore_index=True)

    p1["action_pos"] = p1_position
    p1["action_sel"] = p1_selected
    p2["action_pos"] = p2_position
    p1["action_sel"] = p1_selected

    # Función de recompensa
    p1["reward"] = result
    p2["reward"] = -1 if result == 1 else 1 if result == -1 else 0

    return p1, p2


# ####################################################################
def get_SAR(
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    experiment_name: str,
    number_of_matches: int = 1000,
    steps_per_batch: int = 10_000,
) -> TensorDict:
    """
    steps_per_batch: int = must be greater than ``number_of_matches`` ~ 10x.
    It takes the last ``steps_per_batch`` steps of the matches played.
    """
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

    p_all = pd.DataFrame()
    for match_path, result in results.items():
        logger.debug(f"Processing match: {match_path}, Result: {result}")
        p1, p2 = process_match(match_path, result)

        p_all = pd.concat([p_all, p1], ignore_index=True)
        p_all = pd.concat([p_all, p2], ignore_index=True)

        if len(p_all) >= batch_size:
            break

    p_all = p_all.tail(batch_size)

    p_all = p_all.reset_index(drop=True)
    with set_list_to_stack(False):
        experience = TensorDict(
            {
                "observation": np.stack(p_all["observation"].apply(Board.deserialize)),  # type: ignore
                "action_pos": np.stack(p_all["action_pose"].apply(Board.get_position_index)),  # type: ignore
                "action_sel": np.stack(p_all["action_sel"].apply(Board.get_position_index)),  # type: ignore
                "reward": np.stack(p_all["reward"]),  # type: ignore
            },
            batch_size=[p_all.shape[0]],
        )
    return experience
