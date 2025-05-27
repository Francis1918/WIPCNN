from quartopy import play_games
from bot.CNN_bot import Quarto_bot

player1 = Quarto_bot(
    "C:/Users/Laboratorio IA/Desktop/interpretability_experiments/models/weights/QuartoCNN1/20250527_1315-aqui3.pt"
)
player2 = Quarto_bot()

play_games(
    matches=10,
    player1=player1,
    player2=player2,
)

player1.model
# go_quarto(
#     10,
#     "CNN_bot",
#     "CNN_bot",
#     0,
#     {
#         "model_path": "C:/Users/Laboratorio IA/Desktop/interpretability_experiments/models/weights/QuartoCNN1/20250527_1315-aqui3.pt"
#     },  # No pre-trained model path
#     {"model_path": None},  # No pre-trained model path
#     verbose=True,
#     builtin_bots=False,
# )
