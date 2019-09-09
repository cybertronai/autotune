# Module to hold global variables for curvature computation functions.
# This is needed sincne functionality may be split over several modules

from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

event_writer: Optional[SummaryWriter] = None
token_count: int = 0

debug_dump_stats: bool = False

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
