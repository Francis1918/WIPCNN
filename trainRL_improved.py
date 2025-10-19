from utils.logger import logger

logger.info("Starting. Importing...")

from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from bot.CNN_bot import Quarto_bot
from models.CNN1 import QuartoCNN
from QuartoRL import gen_experience, run_contest

from tqdm.auto import tqdm
import pprint
import pickle
from colorama import init, Fore, Style
from pathlib import Path
import re
import numpy as np

# Bokeh imports for visualization
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row
from bokeh.models import HoverTool, Legend
from bokeh.io import export_png
import pandas as pd

logger.info("Imports done.")

torch.manual_seed(50)
EXPERIMENT_NAME = "improved_training"

# ===========================
# TRAINING DATA DIRECTORY
# ===========================
TRAINING_DATA_DIR = Path(r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\Datos del entrenamiento")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR = TRAINING_DATA_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Training data will be saved to: {TRAINING_DATA_DIR}")
logger.info(f"Model checkpoints will be saved to: {CHECKPOINTS_DIR}")

# ===========================
# IMPROVED HYPERPARAMETERS
# ===========================

DEBUG_PARAMS = False  # Set to True for quick testing

if not DEBUG_PARAMS:
    logger.info("Using improved training parameters.")
    BATCH_SIZE = 256
    
    RIVALS_IN_TOURNAMENT = 100
    N_MATCHES_EVAL = 10
    
    EPOCHS = 100_000
    
    MATCHES_PER_EPOCH = 300
    STEPS_PER_EPOCH = 10 * MATCHES_PER_EPOCH
    ITER_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE
    
    # IMPROVED: Larger replay buffer for better experience diversity
    REPLAY_SIZE = 200 * STEPS_PER_EPOCH  # Doubled from 100 to 200
    
    N_BATCHS_2_UPDATE_TARGET = ITER_PER_EPOCH // 3
    
    N_LAST_STATES_INIT: int = 2
    N_LAST_STATES_FINAL: int = -1
    
    # IMPROVED: Higher exploration temperature for better exploration
    TEMPERATURE_EXPLORE = 1.0  # Increased from 0.5 to 1.0
    TEMPERATURE_EXPLOIT = 0.1
    
    N_PLAYERS_PLOT = 7
    POINTS_BY_RIVAL = 50
    
else:
    logger.warning("DEBUG MODE: Using smaller batch size and fewer epochs for debugging purposes.")
    BATCH_SIZE = 16
    EPOCHS = 1000
    
    ITER_PER_EPOCH = 5
    MATCHES_PER_EPOCH = 10
    STEPS_PER_EPOCH = 100
    
    REPLAY_SIZE = 600  # Doubled from 300
    N_BATCHS_2_UPDATE_TARGET = 30
    N_MATCHES_EVAL = 5
    
    N_LAST_STATES_INIT: int = 2
    N_LAST_STATES_FINAL: int = -1
    TEMPERATURE_EXPLORE = 2.0  # Increased
    TEMPERATURE_EXPLOIT = 0.1
    
    N_PLAYERS_PLOT = 4
    RIVALS_IN_TOURNAMENT = 15
    POINTS_BY_RIVAL = 6

# IMPROVED: Better hyperparameters
MAX_GRAD_NORM = 1.0
LR = 1e-4
TAU = 0.005
GAMMA = 0.99

# IMPROVED: Add weight decay for regularization
WEIGHT_DECAY = 1e-5

# IMPROVED: Learning rate warmup and better scheduling
WARMUP_EPOCHS = 100
MIN_LR = 1e-6

# ===========================
# METRICS TRACKING
# ===========================

class TrainingMetrics:
    """Class to track and visualize training metrics"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.metrics = {
            'epoch': [],
            'loss': [],
            'avg_q_value': [],
            'max_q_value': [],
            'min_q_value': [],
            'grad_norm': [],
            'learning_rate': [],
            'win_rate': [],
            'avg_reward': [],
            'exploration_temp': [],
            'replay_buffer_size': []
        }
        
    def add_metrics(self, epoch, loss, q_values, grad_norm, lr, win_rate=None, 
                   avg_reward=None, exploration_temp=None, buffer_size=None):
        """Add metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['loss'].append(loss)
        self.metrics['avg_q_value'].append(np.mean(q_values))
        self.metrics['max_q_value'].append(np.max(q_values))
        self.metrics['min_q_value'].append(np.min(q_values))
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['learning_rate'].append(lr)
        self.metrics['win_rate'].append(win_rate if win_rate is not None else 0)
        self.metrics['avg_reward'].append(avg_reward if avg_reward is not None else 0)
        self.metrics['exploration_temp'].append(exploration_temp if exploration_temp is not None else 0)
        self.metrics['replay_buffer_size'].append(buffer_size if buffer_size is not None else 0)
        
    def save_metrics(self, training_dir):
        """Save metrics to pickle file"""
        metrics_file = training_dir / f"{self.experiment_name}_metrics.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.metrics, f)
        logger.info(f"Metrics saved to {metrics_file}")
        
    def create_bokeh_visualization(self, training_dir):
        """Create comprehensive Bokeh visualization"""
        html_file = training_dir / f"{self.experiment_name}_training_metrics.html"
        output_file(str(html_file))
        
        # Create figures
        p1 = figure(title="Training Loss", x_axis_label='Epoch', y_axis_label='Loss',
                   width=800, height=400)
        p1.line(self.metrics['epoch'], self.metrics['loss'], 
               line_width=2, color='navy', alpha=0.8, legend_label='Loss')
        p1.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("Loss", "@y")]))
        
        p2 = figure(title="Q-Values", x_axis_label='Epoch', y_axis_label='Q-Value',
                   width=800, height=400)
        p2.line(self.metrics['epoch'], self.metrics['avg_q_value'], 
               line_width=2, color='green', alpha=0.8, legend_label='Avg Q-Value')
        p2.line(self.metrics['epoch'], self.metrics['max_q_value'], 
               line_width=1, color='red', alpha=0.5, legend_label='Max Q-Value')
        p2.line(self.metrics['epoch'], self.metrics['min_q_value'], 
               line_width=1, color='blue', alpha=0.5, legend_label='Min Q-Value')
        p2.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("Q-Value", "@y")]))
        
        p3 = figure(title="Learning Rate", x_axis_label='Epoch', y_axis_label='LR',
                   width=800, height=400)
        p3.line(self.metrics['epoch'], self.metrics['learning_rate'], 
               line_width=2, color='orange', alpha=0.8, legend_label='Learning Rate')
        p3.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("LR", "@y")]))
        
        p4 = figure(title="Win Rate", x_axis_label='Epoch', y_axis_label='Win Rate',
                   width=800, height=400)
        p4.line(self.metrics['epoch'], self.metrics['win_rate'], 
               line_width=2, color='purple', alpha=0.8, legend_label='Win Rate')
        p4.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("Win Rate", "@y")]))
        
        p5 = figure(title="Gradient Norm", x_axis_label='Epoch', y_axis_label='Grad Norm',
                   width=800, height=400)
        p5.line(self.metrics['epoch'], self.metrics['grad_norm'], 
               line_width=2, color='red', alpha=0.8, legend_label='Gradient Norm')
        p5.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("Grad Norm", "@y")]))
        
        p6 = figure(title="Exploration Temperature", x_axis_label='Epoch', y_axis_label='Temperature',
                   width=800, height=400)
        p6.line(self.metrics['epoch'], self.metrics['exploration_temp'], 
               line_width=2, color='brown', alpha=0.8, legend_label='Exploration Temp')
        p6.add_tools(HoverTool(tooltips=[("Epoch", "@x"), ("Temperature", "@y")]))
        
        # Combine all plots
        layout = column(p1, p2, p3, p4, p5, p6)
        save(layout)
        logger.info(f"Bokeh visualization saved to {html_file}")

# ===========================
# CHECKPOINT MANAGEMENT
# ===========================

def find_latest_checkpoint(experiment_name: str) -> tuple[int, str | None]:
    """Find the latest checkpoint for the given experiment name."""
    weights_dir = CHECKPOINTS_DIR / "QuartoCNN1"
    
    if not weights_dir.exists():
        logger.warning(f"Weights directory not found: {weights_dir}")
        return -1, None
    
    pattern = f"*-{experiment_name}_epoch_*.pt"
    checkpoint_files = list(weights_dir.glob(pattern))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found for experiment: {experiment_name}")
        return -1, None
    
    epoch_pattern = re.compile(rf"{experiment_name}_epoch_(\d+)\.pt")
    epochs_and_files = []
    
    for file_path in checkpoint_files:
        match = epoch_pattern.search(file_path.name)
        if match:
            epoch_num = int(match.group(1))
            epochs_and_files.append((epoch_num, str(file_path)))
    
    if not epochs_and_files:
        logger.warning("No valid epoch numbers found in checkpoint files")
        return -1, None
    
    latest_epoch, latest_file = max(epochs_and_files, key=lambda x: x[0])
    
    logger.info(f"Found latest checkpoint: Epoch {latest_epoch} at {latest_file}")
    return latest_epoch, latest_file


def load_epochs_results(experiment_name: str, training_dir: Path) -> list:
    """Load the epochs results from pickle file."""
    results_file = training_dir / f"{experiment_name}.pkl"
    
    if results_file.exists():
        try:
            with open(results_file, "rb") as f:
                epochs_results = pickle.load(f)
            logger.info(f"Loaded {len(epochs_results)} epochs from results file")
            return epochs_results
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            return []
    else:
        logger.warning(f"Results file not found: {results_file}")
        return []


def get_all_checkpoints(experiment_name: str) -> list[str]:
    """Get all checkpoint files sorted by epoch number."""
    weights_dir = CHECKPOINTS_DIR / "QuartoCNN1"
    
    if not weights_dir.exists():
        return []
    
    pattern = f"*-{experiment_name}_epoch_*.pt"
    checkpoint_files = list(weights_dir.glob(pattern))
    
    epoch_pattern = re.compile(rf"{experiment_name}_epoch_(\d+)\.pt")
    epochs_and_files = []
    
    for file_path in checkpoint_files:
        match = epoch_pattern.search(file_path.name)
        if match:
            epoch_num = int(match.group(1))
            epochs_and_files.append((epoch_num, str(file_path)))
    
    epochs_and_files.sort(key=lambda x: x[0])
    
    return [f for _, f in epochs_and_files]

# ===========================
# IMPROVED LEARNING RATE SCHEDULER
# ===========================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

# ===========================
# MAIN TRAINING LOOP
# ===========================

# Load or initialize models
latest_epoch, latest_checkpoint = find_latest_checkpoint(EXPERIMENT_NAME)

policy_net = QuartoCNN()
target_net = QuartoCNN()

# Initialize metrics tracker
metrics_tracker = TrainingMetrics(EXPERIMENT_NAME)

if latest_checkpoint is not None:
    logger.info(f"Resuming training from epoch {latest_epoch}")
    logger.info(f"Loading checkpoint: {latest_checkpoint}")
    
    policy_net.load_state_dict(torch.load(latest_checkpoint))
    target_net.load_state_dict(policy_net.state_dict())
    
    epochs_results = load_epochs_results(EXPERIMENT_NAME, TRAINING_DATA_DIR)
    checkpoints_files = get_all_checkpoints(EXPERIMENT_NAME)
    
    start_epoch = latest_epoch + 1
    
    logger.info(f"Resuming from epoch {start_epoch}")
    logger.info(f"Found {len(checkpoints_files)} existing checkpoints")
    logger.info(f"Found {len(epochs_results)} epochs results")
else:
    logger.info("No checkpoint found. Starting training from scratch.")
    target_net.load_state_dict(policy_net.state_dict())
    epochs_results = []
    checkpoints_files = []
    start_epoch = 0
    
    checkpoint_name_generator = lambda epoch: f"{EXPERIMENT_NAME}_epoch_{epoch:04d}"
    checkpoint_name = checkpoint_name_generator(0)
    _fcheckpoint_name = policy_net.export_model(checkpoint_name, checkpoint_folder=str(CHECKPOINTS_DIR))
    checkpoints_files.append(_fcheckpoint_name)

checkpoint_name_generator = lambda epoch: f"{EXPERIMENT_NAME}_epoch_{epoch:04d}"

# Initialize replay buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# IMPROVED: Optimizer with weight decay
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY)

# IMPROVED: Custom scheduler with warmup
scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, MIN_LR, LR)

# Advance the scheduler to the correct epoch if resuming
if start_epoch > 0:
    for _ in range(start_epoch):
        scheduler.step()
    logger.info(f"Scheduler advanced to epoch {start_epoch}")
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

loss_fcn = nn.SmoothL1Loss()

init(autoreset=True)

total_iterations_remaining = (EPOCHS - start_epoch) * ITER_PER_EPOCH

pbar = tqdm(
    total=total_iterations_remaining,
    desc=f"{Fore.CYAN}\n Update network{Style.RESET_ALL}",
    leave=True,
    position=1,
    unit="Iter.",
)

logger.info("Hyperparameters loaded.")
logger.info(f"Starting training from epoch {start_epoch} to {EPOCHS}...")

# Training loop
for e in tqdm(
    range(start_epoch, EPOCHS),
    desc=f"{Fore.GREEN}Epochs{Style.RESET_ALL}",
    position=1,
    leave=False,
    initial=start_epoch,
    total=EPOCHS
):
    # Load models
    p1 = Quarto_bot(model=policy_net)
    p2 = Quarto_bot(model=policy_net)
    
    # IMPROVED: Dynamic temperature decay
    current_temp = TEMPERATURE_EXPLORE * (1 - e / EPOCHS) + TEMPERATURE_EXPLOIT * (e / EPOCHS)
    
    p1.DETERMINISTIC = False
    p1.TEMPERATURE = current_temp
    p2.DETERMINISTIC = False
    p2.TEMPERATURE = current_temp
    logger.debug(f"Using temperatures: p1={p1.TEMPERATURE}, p2={p2.TEMPERATURE}")
    
    logger.debug("Generating experience for epoch %d", e + 1)
    
    if N_LAST_STATES_FINAL == -1:
        n_last_states = 100
    else:
        if EPOCHS == 1:
            n_last_states = N_LAST_STATES_FINAL if N_LAST_STATES_FINAL != -1 else N_LAST_STATES_INIT
        else:
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
    
    replay_buffer.extend(exp)
    
    # Track metrics for this epoch
    epoch_losses = []
    epoch_q_values = []
    epoch_grad_norms = []
    epoch_rewards = []
    
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
        
        valid_pos_mask = action_pos != -1
        valid_sel_mask = action_sel != -1
        valid_mask = valid_pos_mask & valid_sel_mask
        
        if torch.sum(valid_mask) == 0:
            logger.warning("No valid actions in batch, skipping...")
            continue
        
        state_board = state_board[valid_mask]
        state_piece = state_piece[valid_mask]
        action_pos = action_pos[valid_mask]
        action_sel = action_sel[valid_mask]
        done_batch = done_batch[valid_mask]
        next_state_board = next_state_board[valid_mask]
        next_state_piece = next_state_piece[valid_mask]
        reward = data["reward"][valid_mask]
        
        pred_board_pos, pred_piece = policy_net(state_board, state_piece)
        
        dim_reshape = [-1] + [1] * (pred_piece.dim() - 1)
        state_pos_action_values = pred_board_pos.gather(
             1, action_pos.reshape(dim_reshape).type(torch.int64)
         )
        
        state_sel_action_values = pred_piece.gather(
            1, action_sel.reshape(dim_reshape).type(torch.int64)
        )
        
        valid_batch_size = torch.sum(valid_mask).item()
        next_state_sel_values = torch.zeros(valid_batch_size)
        
        non_final_mask = ~done_batch
        
        with torch.no_grad():
            _, _next_state_piece = target_net(
                next_state_board[non_final_mask], next_state_piece[non_final_mask]
            )
        _v2 = _next_state_piece.max(dim=1).values
        next_state_sel_values[non_final_mask] = _v2
        
        expected_state_action_values = (next_state_sel_values * GAMMA) + reward
        
        loss = loss_fcn(
            state_sel_action_values, expected_state_action_values.unsqueeze(1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(), MAX_GRAD_NORM
        )
        if total_norm > MAX_GRAD_NORM:
            logger.warning(
                f"Gradient clipping activated! Total norm before clipping: {total_norm:.4f}"
            )
        optimizer.step()
        
        # Track metrics
        epoch_losses.append(loss.item())
        epoch_q_values.extend(state_sel_action_values.detach().cpu().numpy().flatten())
        epoch_grad_norms.append(total_norm.item())
        epoch_rewards.extend(reward.cpu().numpy().flatten())
        
        if i % N_BATCHS_2_UPDATE_TARGET == 0:
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
    
    # Run contest
    p1.DETERMINISTIC = True
    p1.TEMPERATURE = TEMPERATURE_EXPLOIT
    contest_results = run_contest(
        player=p1,
        rivals=checkpoints_files[:-1],
        rival_class=Quarto_bot,
        rivals_clip=RIVALS_IN_TOURNAMENT,
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
    
    # Calculate win rate
    total_wins = sum(r["wins"] for r in contest_results.values())
    total_games = sum(r["wins"] + r["losses"] + r["draws"] for r in contest_results.values())
    win_rate = total_wins / total_games if total_games > 0 else 0
    
    # Add metrics to tracker
    current_lr = scheduler.get_last_lr()[0]
    metrics_tracker.add_metrics(
        epoch=e + 1,
        loss=np.mean(epoch_losses) if epoch_losses else 0,
        q_values=epoch_q_values if epoch_q_values else [0],
        grad_norm=np.mean(epoch_grad_norms) if epoch_grad_norms else 0,
        lr=current_lr,
        win_rate=win_rate,
        avg_reward=np.mean(epoch_rewards) if epoch_rewards else 0,
        exploration_temp=current_temp,
        buffer_size=len(replay_buffer)
    )
    
    # Save metrics periodically
    if (e + 1) % 10 == 0:
        metrics_tracker.save_metrics(TRAINING_DATA_DIR)
        metrics_tracker.create_bokeh_visualization(TRAINING_DATA_DIR)
    
    # Update learning rate
    scheduler.step()
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

logger.info("Training completed.")

# Final metrics save and visualization
metrics_tracker.save_metrics(TRAINING_DATA_DIR)
metrics_tracker.create_bokeh_visualization(TRAINING_DATA_DIR)

logger.info(f"Training metrics visualization saved to {TRAINING_DATA_DIR / f'{EXPERIMENT_NAME}_training_metrics.html'}")