"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from .kfac.kfac import KFAC
from .kfacvi.kfac import KFAC_VI
from . import lr_scheduler
from . import lr_scheduler_iter

del kfac.kfac
del kfacvi.kfac
