import torch
import torch.nn as nn
from typing import List
from functools import partial
import subprocess
import re
import os
import time


def nvidia_smi_memory_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.split("\n")[:-1]

    gpu_memory_info = []
    for line in output:
        gpu_id, total_memory, used_memory, free_memory = map(int, re.split(",\s", line))
        gpu_memory_info.append(
            {
                "id": gpu_id,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory,
            }
        )

    return gpu_memory_info


def get_gpu_memory():
    memory_info = []
    gpu_memory_info = nvidia_smi_memory_info()

    for gpu in gpu_memory_info:
        gpu_id = gpu["id"]
        total_memory = gpu["total_memory"]
        used_memory  = gpu["used_memory"]
        free_memory  = gpu["free_memory"]
        memory_info.append((gpu_id, total_memory, used_memory, free_memory))

    assert "CUDA_VISIBLE_DEVICES" in os.environ
    gpu_ids     = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    gpu_info    = []
    for id in gpu_ids:
        gpu_info.append(memory_info[int(id)])

    return gpu_info


def pool_request(timestep:int, lowest_gpu_mem:int):
    
    while True:
        gpu_infos = get_gpu_memory()

        symble = 1
        for gpu_info in gpu_infos:
            free_memory     = int(gpu_info[3])
            if free_memory < lowest_gpu_mem:
                symble = 0

        if symble == 1:
            break
        else:
            print("GPU is occupied, please Wait!")
            time.sleep(timestep)  



if __name__ == "__main__":
    info = pool_request(5,17000)
    print(info)