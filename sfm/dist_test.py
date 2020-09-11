import fire
import torch
import pprint
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP

def train(*,
  local_rank=0,
  local_world_size=1,
):
  args = locals()
  pprint.PrettyPrinter(indent=4).pprint(args)
  setup_dist(local_rank, local_world_size)

  ngpus = torch.cuda.device_count() // local_world_size
  device_ids = list(range(local_rank * ngpus, (local_rank + 1) * ngpus))
  print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {ngpus}, device_ids = {device_ids}"
  )
  device = torch.device('cuda', device_ids[0])

  print('training on ' + device.type)

  cleanup_dist()
  # return model


def setup_dist(rank, world_size):
  env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
  }
  # TODO - remove this
  print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
  dist.init_process_group(backend="nccl", init_method='env://')
  print(
      f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
      + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
  )

def cleanup_dist():
    dist.destroy_process_group()

if __name__=='__main__':
  fire.Fire(train)
  