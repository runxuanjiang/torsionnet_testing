import numpy as np
import random
import torch

from torsionnet.utils import *
from torsionnet.agents import A2CRecurrentAgent
from torsionnet.config import Config
from torsionnet.environments import Task
from torsionnet.models import RTGNRecurrent

from torsionnet.generate_molecule import DiffV2

import logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_feature(tag, model):
    mol_config = DiffV2()
    config = Config()
    config.tag = tag

    # Global Settings
    config.network = model
    config.hidden_size = model.dim

    # Batch Hyperparameters
    config.num_workers = 3
    config.rollout_length = 5
    config.recurrence = 5
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2

    # Coefficient Hyperparameters
    lr = 5e-5 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # Task Settings
    config.train_env = Task('ConfEnv-v1', concurrency=False, num_envs=config.num_workers, seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)
    config.eval_env = Task('ConfEnv-v1', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)
    config.curriculum = None

    return A2CRecurrentAgent(config)


if __name__ == '__main__':
    nnet = RTGNRecurrent(6, 128, edge_dim=6, point_dim=5)
    nnet.to(device)
    set_one_thread()
    tag = 'Diff-a2c_v0'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
