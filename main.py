import argparse
import torch
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from algorithms.maddpg import MADDPG
#from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()

def run():
  pass

if __name__ == '__main__':
  config = []

  for _ in range(4):
    config.append({
      "num_in_pol": 1,
      "num_out_pol": 1,
      "num_in_critic": 1,
    })

  maddpg = MADDPG(config, 4, discrete_action=True)
