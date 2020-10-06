# Really simple logger

import datetime

DEBUG = 0
INFO = 1
WARNING = 2
def new_logger(min_level, rank):
  def log(level, *s):
    if rank != 0:
      return
    if level < min_level:
      return
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]:', *s)
  return log