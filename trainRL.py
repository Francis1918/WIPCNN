import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from bot.CNN_bot import Quarto_bot
from models.CNN1 import QuartoCNN
from QuartoRL import get_SAR
from tqdm import trange

torch.manual_seed(15)

# epochs = 100
# matches_per_epoch = 1000
# steps_per_batch = 10_000  # ~x10 of matches_per_epoch
# replay_size = 50_000  # ~x5 last epochs
epochs = 6
steps_per_epoch = 10

matches_per_epoch = 10
steps_per_match = 10_0  # ~x10 of matches_per_epoch
replay_size = 50_0  # ~x5 last epochs

batch_size = 64

# ###########################
max_grad_norm = 1.0
LR = 1e-4
TAU = 0.005

# ###########################
policy_net = QuartoCNN()
target_net = QuartoCNN()
target_net.load_state_dict(policy_net.state_dict())


# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=steps_per_match),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.0)


# The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of Q are very noisy.
loss_fcn = nn.SmoothL1Loss()


# ###########################
for e in trange(epochs, desc="Epochs\n", leave=True):
    p1 = Quarto_bot(model=policy_net)
    p2 = Quarto_bot(model=policy_net)  # self play

    exp = get_SAR(
        p1_bot=p1,
        p2_bot=p2,
        number_of_matches=matches_per_epoch,
        steps_per_batch=steps_per_match,
        experiment_name=f"epoch_{e + 1}",
    )

    replay_buffer.extend(exp)  # type: ignore

    for i in range(steps_per_epoch):
        data = replay_buffer.sample(batch_size)

        loss = loss_fcn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # Optimization: grad clipping and optimization step
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        optimizer.step()
        # optimizer.zero_grad() # in PPO

    # Update the target network
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    if i % 10 == 0:
        pass
    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    print(f"Current learning rate: {scheduler.get_last_lr()[0]}")
