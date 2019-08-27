# module to hold global variables
from typing import Optional

import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

event_writer: Optional[SummaryWriter] = None
token_count: int = 0
backward_idx: int = 0  # used by save_backprops to decide where to save values

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
