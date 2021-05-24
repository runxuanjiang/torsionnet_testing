import numpy as np
import torch

from torsionnet.utils import *
from torsionnet.agents import PPORecurrentAgent
from torsionnet.config import Config
from torsionnet.environments import Task
from torsionnet.models import RTGNRecurrent

from torsionnet.generate_molecule import DIFF, DiffV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

def ppo_feature(tag, model):
    mol_config = DiffV2()
    config = Config()
    config.tag = tag

    # Global Settings
    config.network = model
    config.hidden_size = model.dim

    # Batch Hyperparameters
    config.num_workers = 2
    config.rollout_length = 2
    config.recurrence = 1
    config.optimization_epochs = 1
    config.max_steps = 16
    config.save_interval = 8
    config.eval_interval = 0
    config.eval_episodes = 3
    config.mini_batch_size = 4

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)
    config.discount = 0.9999
    config.use_gae = True
    config.gae_tau = 0.95
    config.value_loss_weight = 0.25 # vf_coef
    config.entropy_weight = 0.001
    config.gradient_clip = 0.5
    config.ppo_ratio_clip = 0.2

    # Task Settings
    config.train_env = Task('ConfEnv-v1', concurrency=False, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=4)
    config.eval_env = Task('ConfEnv-v1', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=20)
    config.curriculum = None

    return PPORecurrentAgent(config)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    nnet = RTGNRecurrent(6, 128, edge_dim=6, point_dim=5)
    nnet.to(device)
    set_one_thread()
    tag = 'TEST'
    agent = ppo_feature(tag=tag, model=nnet)
    agent.run_steps()
