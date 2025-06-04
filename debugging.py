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
    "c:/Users/Jonathan Zea/Documents/GitHub/hierarchical-SAE/partidas_guardadas/delet/20250604_0130/2025-06-04_01-30-15_match019.csv",
    result=0,
)
b
