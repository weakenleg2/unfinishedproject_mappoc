import torch
from algorithms.resource_aware_maddpg import RA_MADDPG

config = []
for _ in range(1):
    config.append({
        "in_dim": 2,
        "out_dim": 1,
        "critic_in": 5,
    })


ra_maddpg = RA_MADDPG(agent_init_params=config, n_agents=1, hidden_dim=3)
obs = [torch.tensor([1, 2], dtype=torch.float).unsqueeze(0)]
n_obs = [torch.tensor([3, 4], dtype=torch.float).unsqueeze(0)]
action = [torch.tensor([1.123, 1, 0], dtype=torch.float).unsqueeze(0)]
reward = [torch.tensor([1.12], dtype=torch.float).unsqueeze(0)]
done = [torch.tensor([1], dtype=torch.float).unsqueeze(0)]
sample = (obs, action, reward, n_obs, done)

ra_maddpg.update(sample, 0)
