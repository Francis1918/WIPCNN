import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from bot.CNN_bot import Quarto_bot
from models.CNN1 import QuartoCNN
from QuartoRL import gen_experience, run_contest

from utils.logger import logger
from tqdm import tqdm
import pprint
import pickle
from colorama import init, Fore, Style

import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode

torch.manual_seed(50)
EXPERIMENT_NAME = "ab_GoBig"

BATCH_SIZE = 1024

# every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
EPOCHS = 100_000

# number of times the network is updated per epoch
MATCHES_PER_EPOCH = 3000
# ~x10 of matches_per_epoch, used to generate experience
STEPS_PER_EPOCH = 10 * MATCHES_PER_EPOCH
ITER_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE

REPLAY_SIZE = 10 * STEPS_PER_EPOCH  # ~x3 STEPS_PER_EPOCH, info from last 3 epochs

# update target network every n batches processed, ~1/3 of ITER_PER_EPOCH
N_BATCHS_2_UPDATE_TARGET = ITER_PER_EPOCH // 3

N_MATCHES_EVAL = 100  # number of matches to evaluate the bot at the end of each epoch for every previous rival


# # # ########################### DEBUG
# # # every epoch experience is generated with a new bot instance, models are saved at the end of each epoch

# BATCH_SIZE = 16
# EPOCHS = 10

# # number of times the network is updated per epoch
# ITER_PER_EPOCH = 5
# MATCHES_PER_EPOCH = 10
# STEPS_PER_EPOCH = 10_0  # ~x10 of matches_per_epoch, used to generate experience

# REPLAY_SIZE = 3_00  # ~x3 STEPS_PER_EPOCH, info from last 3 epochs

# # update target network every n batches processed, ~1/3 of ITER_PER_EPOCH
# N_BATCHS_2_UPDATE_TARGET = 30

# N_MATCHES_EVAL = 5  # number of matches to evaluate the bot at the end of each epoch for every previous rival


# ###########################
MAX_GRAD_NORM = 1.0
LR = 1e-4
TAU = 0.005
GAMMA = 0.99

# ###########################
policy_net = QuartoCNN()
target_net = QuartoCNN()
target_net.load_state_dict(policy_net.state_dict())


checkpoint_name_generator = lambda epoch: f"{EXPERIMENT_NAME}_epoch_{epoch:04d}"

checkpoint_name = checkpoint_name_generator(0)

# list of file names by epoch
_fcheckpoint_name = policy_net.export_model(checkpoint_name)
checkpoints_files: list[str] = [_fcheckpoint_name]

# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, 0.0)


# The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of Q are very noisy.
loss_fcn = nn.SmoothL1Loss()

epochs_results = []  # to store the results of each epoch
# ###########################
init(autoreset=True)

pbar = tqdm(
    total=EPOCHS * ITER_PER_EPOCH,
    desc=f"{Fore.CYAN}\nProgress{Style.RESET_ALL}",
    leave=True,
    unit="Iter.",
)

for e in tqdm(range(EPOCHS), desc=f"{Fore.GREEN}Epochs{Style.RESET_ALL}", leave=False):
    p1 = Quarto_bot(model=policy_net)
    p2 = Quarto_bot(model=policy_net)  # self play
    logger.debug("Generating experience for epoch %d", e + 1)
    exp = gen_experience(
        p1_bot=p1,
        p2_bot=p2,
        number_of_matches=MATCHES_PER_EPOCH,
        steps_per_batch=STEPS_PER_EPOCH,
        experiment_name=f"epoch_{e + 1}",
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

        # pred_board_pos, pred_piece = policy_net(state_board, state_piece)
        _, pred_piece = policy_net(state_board, state_piece)

        # filter -1 actions, because they are not valid actions.
        # se pone, como que si hubiese escogido la acción 0...???
        # reemplazar por max
        action_pos = torch.where(
            action_pos == -1, torch.zeros_like(action_pos), action_pos
        )
        action_sel = torch.where(
            action_sel == -1, torch.zeros_like(action_sel), action_sel
        )

        # se necesita hacer reshape para que gather funcione correctamente
        # gather requiere que el tensor de acciones tenga la misma cantidad de dimensiones que el tensor de valores
        # dim_reshape = [-1] + [1] * (pred_board_pos.dim() - 1)
        dim_reshape = [-1] + [1] * (pred_piece.dim() - 1)
        # toma los valores de las acciones seleccionadas
        # state_pos_action_values = pred_board_pos.gather(
        #     1, action_pos.reshape(dim_reshape).type(torch.int64)  # solo acepta int64...
        # )

        # pred_piece debe tener mismo tamaño que pred_board_pos
        state_sel_action_values = pred_piece.gather(
            1, action_sel.reshape(dim_reshape).type(torch.int64)
        )

        # Prealloc with 0 because final states have 0 value
        # next_state_pos_values = torch.zeros(BATCH_SIZE)
        next_state_sel_values = torch.zeros(BATCH_SIZE)

        # mask for non-final states
        non_final_mask = torch.where(~done_batch)

        with torch.no_grad():
            # _next_state_pos, _
            _, _next_state_piece = target_net(
                next_state_board[non_final_mask], next_state_piece[non_final_mask]
            )
        # OJO: solo se va a usar la segunda cabeza de salida, que es la de la pieza seleccionada
        # _v1 = _next_state_pos.max(dim=1).values
        _v2 = _next_state_piece.max(dim=1).values
        # next_state_pos_values[non_final_mask] = _v1
        next_state_sel_values[non_final_mask] = _v2

        # Compute the expected Q values
        expected_state_action_values = (next_state_sel_values * GAMMA) + data["reward"]

        loss = loss_fcn(
            state_sel_action_values, expected_state_action_values.unsqueeze(1)
        )

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
        # optimizer.zero_grad() # in PPO

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
    _f_fname = policy_net.export_model(_fname)
    checkpoints_files.append(_f_fname)

    # Ignore the last epoch, as it is the
    contest_results = run_contest(
        player=p1,
        rivals=checkpoints_files[:-1],  # rivals are the previous epochs
        rival_class=Quarto_bot,
        matches=N_MATCHES_EVAL,
        verbose=False,
        match_dir=f"./partidas_guardadas/{EXPERIMENT_NAME}/{_fname}/",
    )
    logger.info(f"Contest results after epoch {e + 1}")
    logger.info(pprint.pformat(contest_results))

    epochs_results.append(dict(contest_results))
    with open(f"{EXPERIMENT_NAME}.pkl", "wb") as f:
        pickle.dump(epochs_results, f)

    # Extract win rates for each epoch and each rival
    # Build a win rate dictionary by rival
    win_rate_by_rival = {}
    for epoch_idx, epoch_result in enumerate(epochs_results):
        for rival_name, rival_result in epoch_result.items():
            total = (
                rival_result["wins"] + rival_result["draws"] + rival_result["losses"]
            )
            win_rate = (rival_result["wins"] + rival_result["draws"]) / total
            if rival_name not in win_rate_by_rival:
                win_rate_by_rival[rival_name] = []
            win_rate_by_rival[rival_name].append(win_rate)

    # Plot win rate by rival without halting execution
    plt.figure(1, figsize=(10, 6), clear=True)
    for rival_name, win_rates in win_rate_by_rival.items():
        plt.plot(
            range(rival_name + 1, rival_name + 1 + len(win_rates)),
            win_rates,
            ".:",
            label=f"Rival epoch {rival_name}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Previous Rivals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

# Prevent matplotlib from closing figures at the end of the script
plt.ioff()
plt.show(block=True)
