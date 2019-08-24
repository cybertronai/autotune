from tensorboardX import SummaryWriter

event_writer: SummaryWriter = None
token_count: int = 0
backward_idx: int = 0  # used by save_backprops to decide where to save values
