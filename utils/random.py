import numpy as np
import torch
import random


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_version = torch.__version__
    if torch_version.startswith('1.8'):
        torch.use_deterministic_algorithms(True)
    elif torch_version.startswith('1.7'):
        torch.set_deterministic(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
