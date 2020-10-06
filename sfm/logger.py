# Really simple logger

from datetime import datetime
from functools import partial


LEVEL_DEBUG = 0
LEVEL_INFO = 1
LEVEL_WARNING = 2


class logger():
  def __init__(self, min_level, rank):
    self.min_level = min_level
    self.rank = rank
    
  def DEBUG(self, *s):
    partial(self._log, LEVEL_DEBUG)
  def INFO(self, *s):
    partial(self._log, LEVEL_INFO)
  def WARNING(self, *s):
    partial(self._log, LEVEL_WARNING)

  def _log(self, level, *s):
    if self.rank != 0:
      return
    if level < self.min_level:
      return
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]:', *s)


__LEVEL_NOOP = 1000
class noop(logger):
  def __init__(self):
    super().__init__(__LEVEL_NOOP, 1)