import os
import torch.distributed as dist

# adopted from VILA: https://github.com/NVlabs/VILA/blob/ec7fb2c264920bf004fd9fa37f1ec36ea0942db5/llava/train/utils.py#L42
def mprint(msg):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and dist.is_initialized():
        if rank == 0:
            return print(f"dist {rank} of {world_size} | {msg}")
        else:
            return
    else:
        return print(msg)
