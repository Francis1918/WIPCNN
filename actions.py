import torch
from torchrl.data import OneHot, Composite


# Define action spec for two discrete classes, each with 16 possibilities
action_spec = Composite(
    {
        "piece": OneHot(n=16, shape=(16,)),  # Corregido: el shape debe coincidir con n
        "board_position": OneHot(n=16, shape=(16,)),  # Corregido: el shape debe coincidir con n
    }
)
