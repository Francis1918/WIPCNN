# from models.CNN1 import QuartoCNN
from bot.CNN_bot import Quarto_bot

from quartopy import go_quarto

go_quarto(
    100,
    "CNN_bot",
    "CNN_bot",
    0,
    verbose=True,
    builtin_bots=False,
)
