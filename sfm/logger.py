# Really simple logger

from datetime import datetime

LEVEL_DEBUG = 0
LEVEL_INFO = 1
LEVEL_WARNING = 2


class logger():
  def __init__(self, min_level, rank):
    self.min_level = min_level
    self.rank = rank
    
  def DEBUG(self, *s):
    self._log(LEVEL_DEBUG, *s)
  def INFO(self, *s):
    self._log(LEVEL_INFO, *s)
  def WARNING(self, *s):
    self._log(LEVEL_WARNING, *s)

  def _log(self, level, *s):
    if self.rank != 0:
      return
    if level < self.min_level:
      return
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]:', *s)


noop = logger(LEVEL_DEBUG, 1)