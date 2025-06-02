data:
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([
        200,
        1
    ]), device=cpu, dtype=torch.float32, is_shared=False),
        collector: TensorDict(
            fields={
                traj_ids: Tensor(shape=torch.Size([
            200
        ]), device=cpu, dtype=torch.int64, is_shared=False)
    },
            batch_size=torch.Size([
        200
    ]),
            device=cpu,
            is_shared=False),
        done: Tensor(shape=torch.Size([
        200,
        1
    ]), device=cpu, dtype=torch.bool, is_shared=False),
        next: TensorDict(
            fields={
                done: Tensor(shape=torch.Size([
            200,
            1
        ]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([
            200,
            3
        ]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([
            200,
            1
        ]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([
            200,
            1
        ]), device=cpu, dtype=torch.int64, is_shared=False),
                terminated: Tensor(shape=torch.Size([
            200,
            1
        ]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([
            200,
            1
        ]), device=cpu, dtype=torch.bool, is_shared=False)
    },
            batch_size=torch.Size([
        200
    ]),
            device=cpu,
            is_shared=False),
        observation: Tensor(shape=torch.Size([
        200,
        3
    ]), device=cpu, dtype=torch.float32, is_shared=False),
        step_count: Tensor(shape=torch.Size([
        200,
        1
    ]), device=cpu, dtype=torch.int64, is_shared=False),
        terminated: Tensor(shape=torch.Size([
        200,
        1
    ]), device=cpu, dtype=torch.bool, is_shared=False),
        truncated: Tensor(shape=torch.Size([
        200,
        1
    ]), device=cpu, dtype=torch.bool, is_shared=False)
},
    batch_size=torch.Size([
    200
]),
    device=cpu,
    is_shared=False)
len(collector)
10