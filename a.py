from torchrl.collectors import SyncDataCollector
import torch
from tensordict import TensorDict

# Ejemplo de TensorDict (estructura de datos)
example_tensordict = TensorDict(
    {
        "action": torch.zeros(200, 1, dtype=torch.float32),
        "collector": TensorDict(
            {
                "traj_ids": torch.zeros(200, dtype=torch.int64),
            },
            batch_size=torch.Size([200]),
        ),
        "done": torch.zeros(200, 1, dtype=torch.bool),
        "next": TensorDict(
            {
                "done": torch.zeros(200, 1, dtype=torch.bool),
                "observation": torch.zeros(200, 3, dtype=torch.float32),
                "reward": torch.zeros(200, 1, dtype=torch.float32),
                "step_count": torch.zeros(200, 1, dtype=torch.int64),
                "terminated": torch.zeros(200, 1, dtype=torch.bool),
                "truncated": torch.zeros(200, 1, dtype=torch.bool),
            },
            batch_size=torch.Size([200]),
        ),
        "observation": torch.zeros(200, 3, dtype=torch.float32),
        "step_count": torch.zeros(200, 1, dtype=torch.int64),
        "terminated": torch.zeros(200, 1, dtype=torch.bool),
        "truncated": torch.zeros(200, 1, dtype=torch.bool),
    },
    batch_size=torch.Size([200]),
)


# Ejemplo de entorno personalizado
def custom_env_step(action):
    # ...your env logic...
    obs = torch.rand(3)  # observation de ejemplo
    reward = torch.tensor([1.0])  # reward de ejemplo
    done = torch.tensor([False])  # done flag de ejemplo
    custom_data = torch.tensor([42.0])  # custom data de ejemplo
    return TensorDict(
        {
            "observation": obs,
            "reward": reward,
            "done": done,
            "custom_data": custom_data,  # campo personalizado
        },
        batch_size=[],
    )


# Ejemplo de uso del colector (código comentado ya que requiere definiciones adicionales)
"""
# Definir funciones necesarias para el colector
def your_env_fn():
    # Función que crea y devuelve tu entorno
    return YourEnvironment()

def your_policy(tensordict):
    # Política que toma un TensorDict y devuelve acciones
    return TensorDict({"action": torch.rand(1)}, batch_size=[])

# Crear el colector
collector = SyncDataCollector(
    create_env_fn=your_env_fn,  # devuelve entornos que producen TensorDicts personalizados
    policy=your_policy,
    total_frames=10000,
    frames_per_batch=200,
)

# Iterar sobre los lotes
for batch in collector:
    # batch es un TensorDict con tus campos personalizados
    print(batch["custom_data"])
    # ...código existente...
"""
