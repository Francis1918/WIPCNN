"""
trainRLResumeCopy.py - Entrenamiento RL con capacidad de REANUDAR desde último checkpoint

Este archivo es idéntico a trainRLCopy.py pero detecta automáticamente el último
checkpoint guardado y continúa el entrenamiento desde ahí.

Uso:
    - Si no hay checkpoints previos: Inicia desde época 0
    - Si hay checkpoints: Carga el último y continúa desde esa época
"""


from utils.logger import logger

logger.info("Starting trainRLResumeCopy - Resume Training Mode...")

from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from bot.CNN_bot import Quarto_bot
from bot.random_bot import Quarto_random_bot
from models.CNN1 import QuartoCNN
from QuartoRL import gen_experience, run_contest

from tqdm.auto import tqdm
import pprint
import pickle
from colorama import init, Fore, Style
from pathlib import Path
import re
import glob

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger.info("Imports done.")


# ===========================
# TRAINING MONITOR WITH PLOTLY
# ===========================
class TrainingMonitor:
    """Monitor de entrenamiento con gráficas interactivas usando Plotly."""

    def __init__(self, max_points: int = 10000, save_dir: Path = None):
        self.max_points = max_points
        self.losses = deque(maxlen=max_points)
        self.loss_steps = deque(maxlen=max_points)
        self.win_rate_random = []  # Win rate vs oponente aleatorio
        self.win_rate_weak = []    # Win rate vs oponente débil (época 0)
        self.epochs = []
        self.step_counter = 0
        self.save_dir = save_dir

    def update_loss(self, loss_value: float):
        """Registra el valor de pérdida en cada iteración."""
        self.step_counter += 1
        self.losses.append(loss_value)
        self.loss_steps.append(self.step_counter)

    def update_win_rates(self, epoch: int, wr_random: float, wr_weak: float):
        """Registra los win rates vs oponentes de referencia."""
        self.epochs.append(epoch)
        self.win_rate_random.append(wr_random)
        self.win_rate_weak.append(wr_weak)

    def save_state(self, filename: str = "training_monitor_state.pkl"):
        """Guarda el estado del monitor para poder reanudar."""
        if self.save_dir is None:
            return

        state = {
            "losses": list(self.losses),
            "loss_steps": list(self.loss_steps),
            "win_rate_random": self.win_rate_random,
            "win_rate_weak": self.win_rate_weak,
            "epochs": self.epochs,
            "step_counter": self.step_counter,
            "max_points": self.max_points
        }

        filepath = self.save_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"TrainingMonitor state saved to: {filepath}")

    def load_state(self, filename: str = "training_monitor_state.pkl") -> bool:
        """Carga el estado previo del monitor. Retorna True si se cargó exitosamente."""
        if self.save_dir is None:
            return False

        filepath = self.save_dir / filename
        if not filepath.exists():
            logger.info("   No se encontró estado previo del monitor de entrenamiento.")
            return False

        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.losses = deque(state["losses"], maxlen=self.max_points)
            self.loss_steps = deque(state["loss_steps"], maxlen=self.max_points)
            self.win_rate_random = state["win_rate_random"]
            self.win_rate_weak = state["win_rate_weak"]
            self.epochs = state["epochs"]
            self.step_counter = state["step_counter"]

            logger.info(f"   📊 Monitor cargado: {len(self.epochs)} épocas, {self.step_counter} iteraciones")
            return True
        except Exception as e:
            logger.warning(f"   Error al cargar estado del monitor: {e}")
            return False

    def plot(self, save_html: bool = True, filename: str = "training_monitor.html"):
        """Genera y guarda la gráfica interactiva de Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "📉 Pérdida durante Entrenamiento (debe BAJAR)",
                "📈 Win Rate vs Oponentes de Referencia (debe SUBIR)"
            ),
            vertical_spacing=0.15
        )

        # Gráfica de pérdida (debe BAJAR)
        if len(self.losses) > 0:
            fig.add_trace(
                go.Scatter(
                    x=list(self.loss_steps),
                    y=list(self.losses),
                    mode='lines',
                    name='Loss',
                    line=dict(color='red', width=1),
                    hovertemplate='Iteración: %{x}<br>Loss: %{y:.6f}<extra></extra>'
                ),
                row=1, col=1
            )

        # Win Rate vs Random (debe SUBIR)
        if len(self.win_rate_random) > 0:
            fig.add_trace(
                go.Scatter(
                    x=self.epochs,
                    y=self.win_rate_random,
                    mode='lines+markers',
                    name='vs Aleatorio',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6),
                    hovertemplate='Época: %{x}<br>Win Rate: %{y:.2%}<extra></extra>'
                ),
                row=2, col=1
            )

        # Win Rate vs Oponente Débil (debe SUBIR)
        if len(self.win_rate_weak) > 0:
            fig.add_trace(
                go.Scatter(
                    x=self.epochs,
                    y=self.win_rate_weak,
                    mode='lines+markers',
                    name='vs Modelo Inicial (Época 0)',
                    line=dict(color='green', width=2),
                    marker=dict(size=6),
                    hovertemplate='Época: %{x}<br>Win Rate: %{y:.2%}<extra></extra>'
                ),
                row=2, col=1
            )

        # Línea de referencia 50% win rate
        if len(self.epochs) > 0:
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="gray",
                row=2, col=1,
                annotation_text="50% (neutral)",
                annotation_position="right"
            )

        fig.update_xaxes(title_text="Iteración", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_xaxes(title_text="Época", row=2, col=1)
        fig.update_yaxes(title_text="Win Rate", range=[0, 1], row=2, col=1)

        fig.update_layout(
            title="🎮 Monitor de Entrenamiento RL - Quarto",
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )

        if save_html and self.save_dir:
            filepath = self.save_dir / filename
            fig.write_html(str(filepath))
            logger.debug(f"Training monitor saved to: {filepath}")

        return fig


def evaluar_vs_random(player_bot: Quarto_bot, n_partidas: int = 10) -> float:
    """Evalúa el bot contra un oponente aleatorio usando play_games de quartopy."""
    from quartopy import play_games
    import tempfile

    random_bot = Quarto_random_bot()

    # Usar directorio temporal para partidas de evaluación (no necesitamos guardarlas)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Jugar mitad de partidas como P1 y mitad como P2
        results_p1 = play_games(
            matches=n_partidas // 2,
            player1=player_bot,
            player2=random_bot,
            verbose=False,
            match_dir=temp_dir,
            return_file_paths=False,
        )

        results_p2 = play_games(
            matches=n_partidas // 2,
            player1=random_bot,
            player2=player_bot,
            verbose=False,
            match_dir=temp_dir,
            return_file_paths=False,
        )

    # Calcular victorias totales del player_bot
    wins = results_p1["P1"] + results_p2["P2"]
    draws = results_p1["Empates"] + results_p2["Empates"]
    total = n_partidas

    return (wins + 0.5 * draws) / total


def evaluar_vs_debil(player_bot: Quarto_bot, checkpoint_debil: str, n_partidas: int = 10) -> float:
    """Evalúa el bot contra el modelo de la época 0 (débil)."""
    import tempfile

    # Usar directorio temporal para partidas de evaluación
    with tempfile.TemporaryDirectory() as temp_dir:
        results = run_contest(
            player=player_bot,
            rivals=[checkpoint_debil],
            rival_class=Quarto_bot,
            rivals_clip=1,
            matches=n_partidas,
            verbose=False,
            match_dir=temp_dir,
        )

    for _, stats in results.items():
        total = stats["wins"] + stats["draws"] + stats["losses"]
        if total > 0:
            return (stats["wins"] + 0.5 * stats["draws"]) / total
    return 0.0


torch.manual_seed(50)
EXPERIMENT_NAME = "ba_increasing_n_last_states"

# ===========================
# TRAINING DATA DIRECTORY
# ===========================
TRAINING_DATA_DIR = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\DatosEntrenamientoDev")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = TRAINING_DATA_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Training data will be saved to: {TRAINING_DATA_DIR}")
logger.info(f"Model checkpoints will be saved to: {CHECKPOINTS_DIR}")


# if True, will use smaller batch size and fewer epochs for debugging
DEBUG_PARAMS = False  # Real training parameters
# DEBUG_PARAMS = True  # Debugging parameters

if not DEBUG_PARAMS:
    logger.info("Using real training parameters.")
    BATCH_SIZE = 256

    RIVALS_IN_TOURNAMENT = 100  # number of rivals to evaluate the bot against in the contest at the end of each epoch
    N_MATCHES_EVAL = 10  # number of matches to evaluate the bot at the end of each epoch for the selected previous rival

    # every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
    EPOCHS = 100_000

    MATCHES_PER_EPOCH = 300  # number self-play matches per epoch
    # ~x10 of matches_per_epoch, used to generate experience
    STEPS_PER_EPOCH = 10 * MATCHES_PER_EPOCH
    # number of times the network is updated per epoch
    ITER_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE

    # ~x100 STEPS_PER_EPOCH, info from last epochs
    REPLAY_SIZE = 100 * STEPS_PER_EPOCH

    # update target network every n batches processed, ~x3/epoch
    N_BATCHS_2_UPDATE_TARGET = ITER_PER_EPOCH // 3

    # number of last states to consider in the experience generation at the beginning of training
    N_LAST_STATES_INIT: int = 2
    # number of last states to consider in the experience generation at the end of training. -1 means all states
    N_LAST_STATES_FINAL: int = -1

    #-----inicio de la modificacion--- FASE 3: Temperatura de exploración más conservadora
    # Valores anteriores:
    # TEMPERATURE_EXPLORE = 0.5
    # TEMPERATURE_EXPLOIT = 0.1
    # temperature for exploration, higher values lead to more exploration
    TEMPERATURE_EXPLORE = 0.3  # Menor temperatura = menos exploración aleatoria
    # temperature for exploitation, lower values lead to more exploitation
    TEMPERATURE_EXPLOIT = 0.05  # Más determinístico en evaluación
    #-----fin de la modificacion---

    # number of players to plot in the win rate graph, -1 means all players
    N_PLAYERS_PLOT = 7

    # number of rival points to plot for each player in the win rate graph
    POINTS_BY_RIVAL = 50  # must be less than or equal to RIVALS_IN_TOURNAMENT

    # Evaluación vs oponentes de referencia (para TrainingMonitor)
    N_MATCHES_EVAL_REFERENCE = 20  # partidas vs random y vs modelo inicial
    MONITOR_UPDATE_FREQ = 5  # actualizar gráfica cada N épocas


else:
    logger.warning(
        "DEBUG MODE: Using smaller batch size and fewer epochs for debugging purposes."
    )
    # ########################### DEBUG

    # every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
    BATCH_SIZE = 16
    EPOCHS = 1000

    # number of times the network is updated per epoch
    ITER_PER_EPOCH = 5
    MATCHES_PER_EPOCH = 10
    STEPS_PER_EPOCH = 100  # ~x10 of matches_per_epoch, used to generate experience

    REPLAY_SIZE = 300  # ~x3 STEPS_PER_EPOCH, info from last 3 epochs

    # update target network every n batches processed, ~1/3 of ITER_PER_EPOCH
    N_BATCHS_2_UPDATE_TARGET = 30

    N_MATCHES_EVAL = 5  # number of matches to evaluate the bot at the end of each epoch for the selected previous rival

    # number of last states to consider in the experience generation at the beginning of training
    N_LAST_STATES_INIT: int = 2
    # number of last states to consider in the experience generation at the end of training. -1 means all states
    N_LAST_STATES_FINAL: int = -1
    # temperature for exploration, higher values lead to more exploration
    TEMPERATURE_EXPLORE = 2

    # temperature for exploitation, lower values lead to more exploitation
    TEMPERATURE_EXPLOIT = 0.1

    N_PLAYERS_PLOT = 4

    RIVALS_IN_TOURNAMENT = 15  # number of rivals to evaluate the bot against in the contest at the end of each epoch
    POINTS_BY_RIVAL = 6

    # Evaluación vs oponentes de referencia (para TrainingMonitor)
    N_MATCHES_EVAL_REFERENCE = 10  # partidas vs random y vs modelo inicial
    MONITOR_UPDATE_FREQ = 2  # actualizar gráfica cada N épocas


# ###########################
MAX_GRAD_NORM = 1.0
LR = 1e-4
TAU = 0.005
GAMMA = 0.99

# ###########################
# FUNCIÓN PARA DETECTAR ÚLTIMO CHECKPOINT
# ###########################
def detectar_ultimo_checkpoint(training_data_dir: Path, experiment_name: str) -> tuple[int, str | None, list[str]]:
    """
    Detecta el último checkpoint guardado y retorna la época y ruta.

    Busca en: TRAINING_DATA_DIR/checkpoints/QuartoCNN1/

    Returns:
        tuple: (última_época, ruta_checkpoint, lista_checkpoints)
               Si no hay checkpoints, retorna (0, None, [])
    """
    # Ruta donde se guardan los checkpoints
    checkpoints_base = training_data_dir / "checkpoints" / "QuartoCNN1"

    logger.info(f"📁 Buscando checkpoints en: {checkpoints_base}")

    # Buscar checkpoints con el patrón del experimento (con posible prefijo de timestamp)
    # Patrón: *experiment_name_epoch_*.pt (el * inicial captura cualquier prefijo como timestamp)
    pattern = checkpoints_base / f"*{experiment_name}_epoch_*.pt"
    checkpoints = glob.glob(str(pattern))

    # También buscar .pth por si acaso
    pattern_pth = checkpoints_base / f"*{experiment_name}_epoch_*.pth"
    checkpoints += glob.glob(str(pattern_pth))

    if not checkpoints:
        logger.info("   No se encontraron checkpoints previos. Iniciando desde época 0.")
        return 0, None, []

    # Ordenar por número de época
    def extraer_epoca(path: str) -> int:
        match = re.search(r'epoch_(\d+)', path)
        return int(match.group(1)) if match else 0

    checkpoints_sorted = sorted(checkpoints, key=extraer_epoca)
    ultima_epoca = extraer_epoca(checkpoints_sorted[-1])
    ultimo_checkpoint = checkpoints_sorted[-1]

    logger.info(f"   ✅ Encontrados {len(checkpoints)} checkpoints.")
    logger.info(f"   Último checkpoint: época {ultima_epoca} -> {Path(ultimo_checkpoint).name}")

    return ultima_epoca, ultimo_checkpoint, checkpoints_sorted


# Detectar checkpoints existentes
START_EPOCH, ULTIMO_CHECKPOINT, checkpoints_files = detectar_ultimo_checkpoint(
    TRAINING_DATA_DIR, EXPERIMENT_NAME
)

# ###########################
# INICIALIZAR MODELOS
# ###########################
policy_net = QuartoCNN()
target_net = QuartoCNN()

checkpoint_name_generator = lambda epoch: f"{EXPERIMENT_NAME}_epoch_{epoch:04d}"

# Cargar desde checkpoint si existe
if ULTIMO_CHECKPOINT is not None:
    logger.info(f"🔄 REANUDANDO entrenamiento desde época {START_EPOCH}")
    logger.info(f"   Cargando checkpoint: {Path(ULTIMO_CHECKPOINT).name}")

    # Cargar pesos del modelo
    policy_net.load_state_dict(torch.load(ULTIMO_CHECKPOINT))
    target_net.load_state_dict(policy_net.state_dict())

    # El checkpoint de referencia (época 0) para evaluación
    # Buscar con patrón que incluya posible prefijo de timestamp
    checkpoint_epoca_0_pattern = TRAINING_DATA_DIR / "checkpoints" / "QuartoCNN1" / f"*{EXPERIMENT_NAME}_epoch_0000.pt"
    checkpoint_epoca_0_matches = glob.glob(str(checkpoint_epoca_0_pattern))

    if checkpoint_epoca_0_matches:
        CHECKPOINT_DEBIL = checkpoint_epoca_0_matches[0]
        logger.info(f"   Checkpoint referencia (época 0): {Path(CHECKPOINT_DEBIL).name}")
    else:
        # Si no existe época 0, usar el primer checkpoint disponible
        CHECKPOINT_DEBIL = checkpoints_files[0] if checkpoints_files else None
        logger.warning(f"   Checkpoint época 0 no encontrado. Usando: {Path(CHECKPOINT_DEBIL).name if CHECKPOINT_DEBIL else 'None'}")

    logger.info(f"✅ Modelo cargado exitosamente. Continuando desde época {START_EPOCH + 1}")
else:
    logger.info("🆕 INICIANDO entrenamiento desde cero (época 0)")
    target_net.load_state_dict(policy_net.state_dict())

    # Guardar checkpoint inicial (época 0)
    checkpoint_name = checkpoint_name_generator(0)
    _fcheckpoint_name = policy_net.export_model(checkpoint_name, checkpoint_folder=str(CHECKPOINTS_DIR))
    checkpoints_files = [_fcheckpoint_name]

    # Referencia fija para evaluación consistente (modelo época 0 - débil)
    CHECKPOINT_DEBIL = _fcheckpoint_name

# Inicializar monitor de entrenamiento con Plotly
training_monitor = TrainingMonitor(max_points=50000, save_dir=TRAINING_DATA_DIR)

# Cargar estado previo del monitor si estamos reanudando
if START_EPOCH > 0:
    training_monitor.load_state()

# Cargar resultados previos si existen
epochs_results = []
results_file = TRAINING_DATA_DIR / f"{EXPERIMENT_NAME}.pkl"
if results_file.exists() and START_EPOCH > 0:
    try:
        with open(results_file, "rb") as f:
            epochs_results = pickle.load(f)
        logger.info(f"📊 Cargados {len(epochs_results)} resultados de épocas anteriores")
    except Exception as e:
        logger.warning(f"No se pudieron cargar resultados previos: {e}")
        epochs_results = []

# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

#-----inicio de la modificacion--- FASE 4: Learning Rate Scheduler mejorado
# Valor anterior: scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0.0)
# Opción 1: StepLR - Reduce LR cada 200 épocas por factor 0.5
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# Opción 2: CosineAnnealing con reinicio (más suave, recomendado)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=200,      # Reiniciar cada 200 épocas
    T_mult=2,     # Duplicar período después de cada reinicio
    eta_min=1e-6  # LR mínimo
)

# Avanzar el scheduler si estamos reanudando
if START_EPOCH > 0:
    for _ in range(START_EPOCH):
        scheduler.step()
    logger.info(f"📈 Scheduler avanzado a época {START_EPOCH}. LR actual: {scheduler.get_last_lr()[0]:.2e}")
#-----fin de la modificacion---


# The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of Q are very noisy.
loss_fcn = nn.SmoothL1Loss()

# epochs_results ya fue inicializado arriba (cargado de archivo si existe)
# ###########################
init(autoreset=True)

# Calcular épocas restantes
EPOCHS_RESTANTES = EPOCHS - START_EPOCH
ITER_TOTAL_RESTANTES = EPOCHS_RESTANTES * ITER_PER_EPOCH

pbar = tqdm(
    total=ITER_TOTAL_RESTANTES,
    desc=f"{Fore.CYAN}\n Update network{Style.RESET_ALL}",
    leave=True,
    position=1,
    unit="Iter.",
)

logger.info("Hyperparameters loaded.")
if START_EPOCH > 0:
    logger.info(f"🔄 REANUDANDO entrenamiento desde época {START_EPOCH + 1} hasta {EPOCHS}")
    logger.info(f"   Épocas restantes: {EPOCHS_RESTANTES}")
else:
    logger.info("🆕 INICIANDO entrenamiento desde época 1 hasta {EPOCHS}")
logger.info("Starting training...")

#-----inicio de la modificacion--- FASE 4: Configuración para mezcla de oponentes
# Valor anterior: No existía esta configuración
import random
MIXED_OPPONENTS_START_EPOCH = 10  # Comenzar a mezclar oponentes después de N épocas
SELF_PLAY_PROBABILITY = 0.5  # 50% self-play, 50% contra oponentes anteriores
#-----fin de la modificacion---

# Loop principal: desde START_EPOCH hasta EPOCHS
for e in tqdm(
    range(START_EPOCH, EPOCHS), desc=f"{Fore.GREEN}Epochs (desde {START_EPOCH}){Style.RESET_ALL}", position=1, leave=False
):
    # load models
    p1 = Quarto_bot(model=policy_net)

    #-----inicio de la modificacion--- FASE 4: Mezclar oponentes
    # Valor anterior: p2 = Quarto_bot(model=policy_net) siempre (solo self-play)
    # Decidir si usar self-play o un oponente de época anterior
    use_self_play = True

    if e >= MIXED_OPPONENTS_START_EPOCH and len(checkpoints_files) > 1:
        use_self_play = random.random() < SELF_PLAY_PROBABILITY

    if use_self_play:
        # Self-play: p2 usa el mismo modelo que p1
        p2 = Quarto_bot(model=policy_net)
        logger.debug(f"Epoch {e+1}: Using self-play")
    else:
        # Cargar un oponente de una época anterior aleatoria
        # Preferir oponentes más recientes (últimos 50% de checkpoints disponibles)
        n_checkpoints = len(checkpoints_files)
        recent_start = max(0, n_checkpoints // 2)  # Últimos 50%
        opponent_checkpoint = random.choice(checkpoints_files[recent_start:])

        # Crear modelo para el oponente y cargar pesos
        opponent_model = QuartoCNN()
        #-----inicio de la modificacion--- CORRECCION: Usar torch.load y load_state_dict
        # Valor anterior: opponent_model.load_model(opponent_checkpoint) - método no existía
        opponent_model.load_state_dict(torch.load(opponent_checkpoint))
        #-----fin de la modificacion---
        p2 = Quarto_bot(model=opponent_model)
        logger.debug(f"Epoch {e+1}: Using opponent from checkpoint: {opponent_checkpoint}")
    #-----fin de la modificacion---

    # modify the bots to use different temperatures for exploration and exploitation
    p1.DETERMINISTIC = False
    p1.TEMPERATURE = TEMPERATURE_EXPLORE
    p2.DETERMINISTIC = False
    p2.TEMPERATURE = TEMPERATURE_EXPLORE
    logger.debug(f"Using temperatures: p1={p1.TEMPERATURE}, p2={p2.TEMPERATURE}")

    logger.debug("Generating experience for epoch %d", e + 1)

    if N_LAST_STATES_FINAL == -1:
        n_last_states = 100  # inf
    else:
        # Handle division by zero risk when EPOCHS = 1
        if EPOCHS == 1:
            # For single epoch training, use final value directly
            n_last_states = N_LAST_STATES_FINAL if N_LAST_STATES_FINAL != -1 else N_LAST_STATES_INIT
        else:
            # Linearly interpolate n_last_states from N_LAST_STATES_INIT to N_LAST_STATES_FINAL over EPOCHS
            n_last_states = round(
                N_LAST_STATES_INIT
                + (N_LAST_STATES_FINAL - N_LAST_STATES_INIT) * (e / (EPOCHS - 1))
            )

    exp = gen_experience(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=MATCHES_PER_EPOCH,
        steps_per_batch=STEPS_PER_EPOCH,
        experiment_name=f"epoch_{e + 1}",
        match_dir=str(TRAINING_DATA_DIR / "partidas_guardadas" / f"epoch_{e + 1}"),
        PROGRESS_MESSAGE=f"{Fore.YELLOW}Generating experience for epoch {e + 1}{Style.RESET_ALL}",
    )

    replay_buffer.extend(exp)  # type: ignore

    for i in range(ITER_PER_EPOCH):
        pbar.update(1)
        data = replay_buffer.sample(BATCH_SIZE)
        if data.shape[0] < BATCH_SIZE:
            logger.warning(
                f"Not enough data to sample a full batch. Expected {BATCH_SIZE}, got {data.shape[0]}"
            )
            continue

        state_board = data["state_board"]
        state_piece = data["state_piece"]
        action_pos = data["action_pos"]
        action_sel = data["action_sel"]
        done_batch = data["done"]
        next_state_board = data["next_state_board"]
        next_state_piece = data["next_state_piece"]

        pred_board_pos, pred_piece = policy_net(state_board, state_piece)

        # Filter out experiences with invalid actions (-1) instead of replacing them with 0
        # This prevents false signals in training
        valid_pos_mask = action_pos != -1
        valid_sel_mask = action_sel != -1
        valid_mask = valid_pos_mask & valid_sel_mask

        # Skip this batch if no valid actions
        if torch.sum(valid_mask) == 0:
            logger.warning("No valid actions in batch, skipping...")
            continue

        # Filter all data to only include valid actions
        state_board = state_board[valid_mask]
        state_piece = state_piece[valid_mask]
        action_pos = action_pos[valid_mask]
        action_sel = action_sel[valid_mask]
        done_batch = done_batch[valid_mask]
        next_state_board = next_state_board[valid_mask]
        next_state_piece = next_state_piece[valid_mask]
        reward = data["reward"][valid_mask]

        # Recalculate predictions for filtered data
        pred_board_pos, pred_piece = policy_net(state_board, state_piece)

        # se necesita hacer reshape para que gather funcione correctamente
        # gather requiere que el tensor de acciones tenga la misma cantidad de dimensiones que el tensor de valores
        dim_reshape = [-1] + [1] * (pred_piece.dim() - 1)
        # toma los valores de las acciones seleccionadas
        state_pos_action_values = pred_board_pos.gather(
             1, action_pos.reshape(dim_reshape).type(torch.int64)  # solo acepta int64...
         )

        # pred_piece debe tener mismo tamaño que pred_board_pos
        state_sel_action_values = pred_piece.gather(
            1, action_sel.reshape(dim_reshape).type(torch.int64)
        )

        # Prealloc with 0 because final states have 0 value - adjusted for filtered data
        valid_batch_size = torch.sum(valid_mask).item()
        next_state_sel_values = torch.zeros(valid_batch_size)

        # mask for non-final states in filtered data - use boolean mask directly
        non_final_mask = ~done_batch

        with torch.no_grad():
            _, _next_state_piece = target_net(
                next_state_board[non_final_mask], next_state_piece[non_final_mask]
            )
        # OJO: solo se va a usar la segunda cabeza de salida, que es la de la pieza seleccionada
        _v2 = _next_state_piece.max(dim=1).values
        next_state_sel_values[non_final_mask] = _v2

        # Compute the expected Q values using filtered rewards
        expected_state_action_values = (next_state_sel_values * GAMMA) + reward

        loss = loss_fcn(
            state_sel_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Actualizar monitor de entrenamiento con el loss
        training_monitor.update_loss(loss.item())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # Optimization: grad clipping and optimization step
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        total_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(), MAX_GRAD_NORM
        )
        if total_norm > MAX_GRAD_NORM:
            logger.warning(
                f"Gradient clipping activated! Total norm before clipping: {total_norm:.4f}"
            )
        optimizer.step()
#optimizar zero.grad en caso que se use en POO
        if i % N_BATCHS_2_UPDATE_TARGET == 0:
            # ----------- Update the target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

    # Save the model at the end of each epoch
    _fname = checkpoint_name_generator(e + 1)
    _f_fname = policy_net.export_model(_fname, checkpoint_folder=str(CHECKPOINTS_DIR))
    checkpoints_files.append(_f_fname)

    # Ignore the last epoch, as it is the current model
    p1.DETERMINISTIC = True
    p1.TEMPERATURE = TEMPERATURE_EXPLOIT
    contest_results = run_contest(
        player=p1,
        rivals=checkpoints_files[:-1],  # rivals are the previous epochs
        rival_class=Quarto_bot,
        rivals_clip=RIVALS_IN_TOURNAMENT,  # limit the number of rivals for evaluation, -1 means no limit
        matches=N_MATCHES_EVAL,
        verbose=False,
        match_dir=str(TRAINING_DATA_DIR / "partidas_guardadas" / EXPERIMENT_NAME / _fname),
        PROGRESS_MESSAGE=f"{Fore.MAGENTA}Running contest for epoch {e + 1}{Style.RESET_ALL}",
    )
    logger.info(f"Contest results after epoch {e + 1}")
    logger.info(pprint.pformat(contest_results))

    epochs_results.append(dict(contest_results))
    results_file = TRAINING_DATA_DIR / f"{EXPERIMENT_NAME}.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(epochs_results, f)

    # ===========================
    # EVALUACIÓN VS OPONENTES DE REFERENCIA (para TrainingMonitor)
    # ===========================
    if (e + 1) % MONITOR_UPDATE_FREQ == 0:
        logger.info(f"Evaluating vs reference opponents at epoch {e + 1}...")

        # Evaluar vs oponente aleatorio
        wr_random = evaluar_vs_random(p1, n_partidas=N_MATCHES_EVAL_REFERENCE)

        # Evaluar vs modelo inicial (débil)
        wr_weak = evaluar_vs_debil(p1, CHECKPOINT_DEBIL, n_partidas=N_MATCHES_EVAL_REFERENCE)

        # Actualizar monitor
        training_monitor.update_win_rates(e + 1, wr_random, wr_weak)

        # Guardar gráfica HTML
        training_monitor.plot(save_html=True, filename="training_monitor.html")

        # Guardar estado del monitor para poder reanudar con las gráficas
        training_monitor.save_state()

        logger.info(f"Win rate vs Random: {wr_random:.2%}, vs Modelo Inicial: {wr_weak:.2%}")

    # Extract win rates for each player epoch and each rival
    win_rate_by_epoch: defaultdict[int, dict[int, float]] = defaultdict(lambda: dict())
    for player_id, player_results in enumerate(epochs_results):
        for player_name, result_vs_rival in player_results.items():
            total = (
                result_vs_rival["wins"]
                + result_vs_rival["draws"]
                + result_vs_rival["losses"]
            )
            win_rate = (
                result_vs_rival["wins"] + 0.5 * result_vs_rival["draws"]
            ) / total

            win_rate_by_epoch[player_id][player_name] = win_rate

    # ===========================
    # PLOTTING CON PLOTLY (Win Rate vs Previous Rivals)
    # ===========================
    # Only plot N_PLAYERS_PLOT equally spaced players (or all if N_PLAYERS_PLOT < 0 or more players than available)
    if N_PLAYERS_PLOT < 0 or N_PLAYERS_PLOT >= e + 1:
        players_to_plot = range(e + 1)  # Plot all players
    else:
        players_to_plot = torch.linspace(0, e, steps=N_PLAYERS_PLOT).long().tolist()

    fig_rivals = go.Figure()

    for player_name in players_to_plot:
        win_rates = win_rate_by_epoch[player_name]
        n = len(win_rates)  # Number of rivals found for this player (epochs)
        if n > POINTS_BY_RIVAL:
            # Select POINTS_BY_RIVAL indices spaced across the whole range
            idx_rival_names = (
                torch.linspace(0, n - 1, steps=POINTS_BY_RIVAL).long().tolist()
            )
            x_rival_names = [list(win_rates.keys())[i] for i in idx_rival_names]
        else:
            # If fewer than POINTS_BY_RIVAL, plot all available points
            x_rival_names = list(win_rates.keys())

        y_win_rates = [win_rates[i] for i in x_rival_names]

        fig_rivals.add_trace(
            go.Scatter(
                x=x_rival_names,
                y=y_win_rates,
                mode='lines+markers',
                name=f"Época {player_name}",
                hovertemplate='Rival época: %{x}<br>Win Rate: %{y:.2%}<extra></extra>'
            )
        )

    fig_rivals.add_hline(y=0.5, line_dash="dash", line_color="gray",
                         annotation_text="50%", annotation_position="right")

    fig_rivals.update_layout(
        title="📊 Win Rate vs Rivales Anteriores",
        xaxis_title="Rival desde época",
        yaxis_title="Win Rate",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=500,
        showlegend=True
    )

    # Guardar gráfica de rivales
    rivals_plot_file = TRAINING_DATA_DIR / "win_rate_vs_rivals.html"
    fig_rivals.write_html(str(rivals_plot_file))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

logger.info("Training completed.")
# Guardar gráfica final del monitor
training_monitor.plot(save_html=True, filename="training_monitor_final.html")
training_monitor.save_state()  # Guardar estado final
logger.info(f"Final training monitor saved to: {TRAINING_DATA_DIR / 'training_monitor_final.html'}")


