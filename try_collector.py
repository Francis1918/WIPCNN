from torchrl.collectors import SyncDataCollector

from tensordict import TensorDict

import torch


def custom_env_step(action):
    obs = ...  # observation
    reward = ...  # reward
    done = ...  # done flag
    custom_data = ...  # any custom data
    return TensorDict(
        {
            "observation": torch.tensor(obs),
            "reward": torch.tensor(reward),
            "done": torch.tensor(done),
            "custom_data": torch.tensor(custom_data),  # your custom field
        },
        batch_size=[],
    )


# Example collector usage
collector = SyncDataCollector(
    create_env_fn=your_env_fn,  # should return envs that output custom TensorDicts
    policy=your_policy,
    total_frames=10000,
    frames_per_batch=200,
)

for batch in collector:
    # batch is a TensorDict with your custom fields
    print(batch["custom_data"])
    # ...existing code...
