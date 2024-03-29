import fire
import torch
import pprint
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def train():
  args = locals()
  pprint.PrettyPrinter(indent=4).pprint(args)
  device_id = setup_dist()
  device = torch.device('cuda', device_id)
  print('training on ' + device.type + ' ' + str(device_id))
  time.sleep(60)
  cleanup_dist()


def setup_dist():
  env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
  }
  # TODO - remove this
  print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
  dist.init_process_group(backend="nccl", init_method='env://')
  print(
      f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
      + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
  )
  device_id =  int(env_dict["LOCAL_RANK"])
  return device_id


def cleanup_dist():
    dist.destroy_process_group()

if __name__=='__main__':
  fire.Fire(train)
  