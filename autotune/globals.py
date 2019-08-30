# Module to hold global variables for curvature computation functions.
# This is needed sincne functionality may be split over several modules, while global variables
# simplify code significantly

from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

event_writer: Optional[SummaryWriter] = None
token_count: int = 0
backward_idx: int = 0  # used by save_backprops to decide where to save values

dataset_root = '/tmp/data'   # where PyTorch datasets will go on auto-download

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


skip_forward_hooks: Optional[bool] = None
skip_backward_hooks: Optional[bool] = None
