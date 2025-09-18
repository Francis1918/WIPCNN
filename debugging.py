# from QuartoRL import gen_experience
# from bot.CNN_bot import Quarto_bot
# import torch

# torch.manual_seed(15)
# p1 = Quarto_bot()
# p2 = Quarto_bot()
# exp = gen_experience(
#     p1_bot=p1,
#     p2_bot=p2,
#     number_of_matches=3,
#     experiment_name="delet",
# )


from QuartoRL import process_match

a, b = process_match(
    "C:/Users/bravo/Documents/Metodos Numericos Pycharm/Mech Interp/hierarchical-SAE/partidas_guardadas/ba_increasing_n_last_states/ba_increasing_n_last_states_epoch_0001/2025-09-14_12-55-49_match001.csv",
    result=0,
)
b
