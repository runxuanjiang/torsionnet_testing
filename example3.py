import numpy as np
import torch

from conformer_rl import utils
from conformer_rl.agents import A2CRecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation import test_alkane

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    mol_config = test_alkane()

    config = Config()
    config.tag = 'example3'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 10
    config.rollout_length = 5
    config.recurrence = 5
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2

    # Coefficient Hyperparameters
    lr = 5e-5* np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScorePruningEnv-v0', concurrency=True, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=200)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=200)
    config.curriculum = None

    agent = A2CRecurrentAgent(config)
    agent.run_steps()