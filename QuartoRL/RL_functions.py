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

try:
    from quartopy import play_games, BotAI, Board
except ImportError:
    # Fallback para ejecución directa o quartopy faltante
    try:
        import sys
        from pathlib import Path

        # Agregar directorio padre al path para setup_dependencies
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Importar y ejecutar configuración de dependencias
        import setup_dependencies

        setup_dependencies.setup_quartopy(silent=False)

        # Reintentar importación después de la configuración
        from quartopy import play_games, BotAI, Board

        logger.info("✅ Quartopy importado exitosamente después de configuración de dependencias")

    except ImportError as e:
        error_msg = (
            "❌ ERROR DE DEPENDENCIA: No se puede importar quartopy\n\n"
            "🔧 PASOS PARA SOLUCIONAR:\n"
            "1. Asegúrate de que el proyecto 'quartopy' esté disponible en tu entorno\n"
            "2. Verifica si quartopy está en alguna de estas ubicaciones:\n"
            "   - ../quartopy (relativo a este proyecto)\n"
            "   - ~/Documents/GitHub/Quartopy\n"
            "   - C:/Users/bravo/Documents/quartopy\n"
            "3. Si quartopy está en otro lugar, crea un archivo .env con:\n"
            "   QUARTOPY_PATH=/ruta/a/tu/proyecto/quartopy\n"
            "4. O instala quartopy como paquete: pip install quartopy\n\n"
            f"📋 Error original: {e}\n\n"
            "💡 Para más ayuda, revisa setup_dependencies.py"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"❌ ERROR INESPERADO durante configuración de quartopy: {e}\n\n"
            "🔧 ACCIONES SUGERIDAS:\n"
            "1. Verifica que setup_dependencies.py existe y es válido\n"
            "2. Comprueba permisos de archivo en el directorio del proyecto\n"
            "3. Intenta ejecutar el proyecto con privilegios de administrador\n"
            "4. Revisa el archivo utils/logger.py por cualquier problema\n\n"
            "💡 Considera agregar quartopy manualmente a tu ruta de Python"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ####################################################################
def process_match(
    match_path: str, result: int, n_last_states: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads a match file and extracts: observation, action, and reward for both players.
    ## Args:
        * match_path (str): Path to the match file.
        * result (int): Result of the match, where 1 is a win for player 1, -1 is a win for player 2, and 0 is a draw.
        * n_last_states (int): Number of last states to consider for the match. Default is 10.
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

    # --- States
    p1["state_board"] = pd.concat(
        [pd.Series(["0"]), df["Tablero"][3::4]], ignore_index=True
    )
    p2["state_board"] = df["Tablero"][1::4].reset_index(drop=True)

    # p1 no recibió pieza en el primer estado, por lo que se pone -1.
    assert (p1.shape[0] - 1) == p2_selected.shape[0]
    p1["state_piece"] = pd.concat([pd.Series([-1]), p2_selected], ignore_index=True)

    # p2 recibió pieza en el primer estado, por lo que se pone la pieza seleccionada.
    assert (p2.shape[0]) == p1_selected.shape[0]
    p2["state_piece"] = p1_selected.reset_index(drop=True)

    # -- next state
    # el estado siguiente de p1 es el estado de p2
    # el estado siguiente de p2 es el estado de p1 pero desde el índice 1.
    if result == 1 or result == 0:
        # p1 gana o empata, hay que añadir un estado vacío en el estado siguiente de p1
        p1["next_state_board"] = pd.concat(
            [p2["state_board"], pd.Series([-1])], ignore_index=True
        )
        p1["next_state_piece"] = pd.concat(
            [p2["state_piece"], pd.Series([-1])], ignore_index=True
        )

        p2["next_state_board"] = p1["state_board"][1:].reset_index(drop=True)
        p2["next_state_piece"] = p1["state_piece"][1:].reset_index(drop=True)
    else:
        # p2 gana
        # ambos jugadores hicieron el mismo número de movimientos
        p1["next_state_board"] = p2["state_board"].reset_index(drop=True)
        # el segundo jugador no seleccionó pieza en el último movimiento, por lo que se pone -1.
        p1["next_state_piece"] = pd.concat(
            [p2["state_piece"], pd.Series([-1])], ignore_index=True
        )

        # no hay siguiente estado siguiente para p2 al finalizar el juego.
        p2["next_state_board"] = pd.concat(
            [p1["state_board"][1:], pd.Series([-1])], ignore_index=True
        )
        p2["next_state_piece"] = pd.concat(
            [p1["state_piece"][1:], pd.Series([-1])], ignore_index=True
        )

    # --- Actions
    # The first move does not put piece in a ``position``. There is no piece to place.
    p1_position = pd.concat([pd.Series([-1]), p1_position], ignore_index=True)

    # Last-winning move does not select a piece, so we add -1 to the end of the selected pieces.
    if result == 1 or result == 0:
        # Cuando p1 gana o empata, no alcanzó a seleccionar pieza.
        p1_selected = pd.concat([p1_selected, pd.Series([-1])], ignore_index=True)
    else:
        # Cuando p2 gana, no alcanzó a seleccionar pieza.
        p2_selected = pd.concat([p2_selected, pd.Series([-1])], ignore_index=True)

    p1["action_pos"] = p1_position.reset_index(drop=True)
    p1["action_sel"] = p1_selected.reset_index(drop=True)
    p2["action_pos"] = p2_position.reset_index(drop=True)
    p2["action_sel"] = p2_selected.reset_index(drop=True)

    # --- End
    p1["done"] = False
    p2["done"] = False

    # Función de recompensa
    p1["reward"] = result
    if result == 1:
        p2["reward"] = -1
        p1.loc[p1.index[-1], "done"] = True

    elif result == -1:
        p2["reward"] = 1
        p2.loc[p2.index[-1], "done"] = True

    else:
        p2["reward"] = 0
        p1.loc[p1.index[-1], "done"] = True

    # --- Last n states
    p1 = p1.tail(n_last_states).reset_index(drop=True)
    p2 = p2.tail(n_last_states).reset_index(drop=True)
    return p1, p2


# ####################################################################
def gen_experience(
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    experiment_name: str,
    n_last_states: int = 10,
    number_of_matches: int = 1000,
    steps_per_batch: int = 10_000,
    verbose: bool = False,
    PROGRESS_MESSAGE: str = "Generating experience...",
) -> TensorDict:
    """
    steps_per_batch: int = must be greater than ``number_of_matches`` ~ 10x.
    It takes the last ``steps_per_batch`` steps of the matches played.
    """
    logger.debug("Generating experience...")

    batch_size = steps_per_batch

    match_dir = f"./partidas_guardadas/{experiment_name}/{datetime.now().strftime('%Y%m%d_%H%M')}/"

    results = play_games(
        matches=number_of_matches,
        player1=p1_bot,
        player2=p2_bot,
        delay=0,
        verbose=verbose,
        match_dir=match_dir,
        PROGRESS_MESSAGE=PROGRESS_MESSAGE,
    )

    logger.debug(
        f"Generated experience. Matches played: {number_of_matches}, Steps per batch: {steps_per_batch}"
    )

    p_all = pd.DataFrame()
    for match_path, result in tqdm(
        results.items(),
        desc="Processing matches",
        mininterval=0.3,
        miniters=20,
        position=0,
        leave=False,
    ):
        logger.debug(f"Processing match: {match_path}, Result: {result}")
        p1, p2 = process_match(match_path, result, n_last_states=n_last_states)

        p_all = pd.concat([p_all, p1], ignore_index=True)
        p_all = pd.concat([p_all, p2], ignore_index=True)

        if len(p_all) >= batch_size:
            break

    p_all = p_all.tail(batch_size)

    p_all = p_all.reset_index(drop=True)
    with set_list_to_stack(False):
        experience = TensorDict(
            {
                "state_board": np.stack(p_all["state_board"].apply(Board.deserialize)),  # type: ignore
                "state_piece": np.stack(p_all["state_piece"].apply(Board.pos_index2vector)),  # type: ignore
                # "action_pos": np.stack(p_all["action_pos"].apply(Board.pos_index2vector)),  # type: ignore
                # "action_sel": np.stack(p_all["action_sel"].apply(Board.pos_index2vector)),  # type: ignore
                "action_pos": np.stack(p_all["action_pos"]),  # type: ignore
                "action_sel": np.stack(p_all["action_sel"]),  # type: ignore
                "reward": np.stack(p_all["reward"]),  # type: ignore
                "done": np.stack(p_all["done"]),  # type: ignore
                "next_state_board": np.stack(
                    p_all["next_state_board"].apply(Board.deserialize)  # type: ignore
                ),
                "next_state_piece": np.stack(
                    p_all["next_state_piece"].apply(Board.pos_index2vector)  # type: ignore
                ),
            },
            batch_size=[p_all.shape[0]],
        )
    return experience
