import torch

torch.manual_seed(15)
from quartopy import go_quarto

go_quarto(
    1_000,
    "CNN_bot",
    "CNN_bot",
    0,
    # params_p1={"model_path": "models/weights/QuartoCNN1\\20250527_1315-aqui3.pt"},
    verbose=True,
    builtin_bots=False,
    # folder_bots="../bot/",
)
