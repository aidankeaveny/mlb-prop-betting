import torch
import random
import numpy as np

TARGET_VAR = 'TotalBases'
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
