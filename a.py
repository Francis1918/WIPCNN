from torchrl.collectors import SyncDataCollector

from tensordict import TensorDict

TensorDict(  # type: ignore
    fields={  # type: ignore
        action: Tensor(  # type: ignore
            shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False  # type: ignore
        ),  # type: ignore
        collector: TensorDict(  # type: ignore
            fields={  # type: ignore
                traj_ids: Tensor(  # type: ignore
                    shape=torch.Size([200]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.int64,  # type: ignore
                    is_shared=False,  # type: ignore
                )  # type: ignore
            },  # type: ignore
            batch_size=torch.Size([200]),  # type: ignore
            device=cpu,  # type: ignore
            is_shared=False,  # type: ignore
        ),  # type: ignore
        done: Tensor(  # type: ignore
            shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False  # type: ignore
        ),  # type: ignore
        next: TensorDict(  # type: ignore
            fields={  # type: ignore
                done: Tensor(  # type: ignore
                    shape=torch.Size([200, 1]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.bool,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
                observation: Tensor(  # type: ignore
                    shape=torch.Size([200, 3]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.float32,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
                reward: Tensor(  # type: ignore
                    shape=torch.Size([200, 1]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.float32,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
                step_count: Tensor(  # type: ignore
                    shape=torch.Size([200, 1]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.int64,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
                terminated: Tensor(  # type: ignore
                    shape=torch.Size([200, 1]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.bool,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
                truncated: Tensor(  # type: ignore
                    shape=torch.Size([200, 1]),  # type: ignore
                    device=cpu,  # type: ignore
                    dtype=torch.bool,  # type: ignore
                    is_shared=False,  # type: ignore
                ),  # type: ignore
            },  # type: ignore
            batch_size=torch.Size([200]),  # type: ignore
            device=cpu,  # type: ignore
            is_shared=False,  # type: ignore
        ),  # type: ignore
        observation: Tensor(  # type: ignore
            shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False  # type: ignore
        ),  # type: ignore
        step_count: Tensor(  # type: ignore
            shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False  # type: ignore
        ),  # type: ignore
        terminated: Tensor(  # type: ignore
            shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False  # type: ignore
        ),  # type: ignore
        truncated: Tensor(  # type: ignore
            shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False  # type: ignore
        ),  # type: ignore
    },  # type: ignore
    batch_size=torch.Size([200]),  # type: ignore
    device=cpu,  # type: ignore
    is_shared=False,  # type: ignore
)  # type: ignore
# type: ignore


# Example custom environment step
def custom_env_step(action):
    # ...your env logic...
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
