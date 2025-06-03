from QuartoRL import get_SAR
from bot.CNN_bot import Quarto_bot
import torch

torch.manual_seed(15)
p1 = Quarto_bot()
p2 = Quarto_bot()
exp = get_SAR(
    p1_bot=p1,
    p2_bot=p2,
    number_of_matches=3,
    experiment_name="delet",
)
