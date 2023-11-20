import gymnasium as gym
from MLPformappo import MLPBase
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F

env = gym.make('Pendulum-v1')
obs_shape = env.observation_space.shape
ac_shape = env.action_space
# import wandb
# wandb.init(project="pend_test", entity="2017920898")
# print(ac_shape)
network = MLPBase(hid_size=96,num_hid_layers=3,num_options=2,dc=0.1
                  ,q_space=obs_shape,ac_space=ac_shape,pi_space=obs_shape,
                  mu_space=obs_shape,num=2)
def compute_loss(memory,gamma=0.99, lambda_gae=0.95,clip_param=0.2,entropy_weight=0.01):
    states, actions, rewards, next_states, dones, vpreds,ac_log = memory.sample(len(memory))

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    print(states)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    vpreds = torch.tensor(vpreds, dtype=torch.float32)
    old_log_probs = torch.tensor(ac_log, dtype=torch.float32)

    # Calculate discounted returns and advantage estimates
    returns = []
    advs = []
    return_ = 0
    adv = 0
    num_steps = len(rewards)
    for step in reversed(range(num_steps)):
        if step == num_steps - 1 or dones[step]:  # Check if this is the last step or a terminal state
            next_value = 0  # No future rewards if the state is terminal or it's the last step
        else:
            next_value = vpreds[step + 1]

        td_error = rewards[step] + gamma * next_value - vpreds[step]
        adv = td_error + gamma * lambda_gae * adv * (1 - dones[step])
        return_ = rewards[step] + gamma * return_ * (1 - dones[step])

        returns.insert(0, return_)
        advs.insert(0, adv)

    returns = torch.tensor(returns, dtype=torch.float32)
    advs = torch.tensor(advs, dtype=torch.float32)
    # print(f"return:{returns}")
    # print(f"adv:{advs}")

    # Policy loss
    # log_probs = network.get_log_probs(states, actions)
    new_log_probs, state_values, entropy = network.evaluate_actions(states, actions)

    # Calculate ratio for PPO
    ratios = torch.exp(new_log_probs - old_log_probs.detach())

    # Calculate clipped surrogate objective
    surr1 = ratios * advs
    surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advs
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(state_values.squeeze(-1), returns)

    # Total loss
    total_loss = policy_loss + value_loss - entropy_weight * entropy.mean()

    return total_loss

total_episodes = 10000
buffer_size = 1000  
memory = ReplayBuffer(buffer_size)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
for episode in range(total_episodes):
    episode = 0
    obs,_ = env.reset()
    # obs = obs[0]
    done = False
    episode_reward = 0

    while not done:
        # Convert observation to tensor
        # obs_tensor = prepare_observation(obs)

        # Sample action from the policy
        # print(obs[0])
        action, vpred, _, ac_log = network.act(stochastic=True, ob=obs, option=None)
        # obs = obs.copy()

        # Execute action in the environment
        next_obs, reward, done, _, _ = env.step(action)

        # Store transition in memory (you need to implement a memory for this)
        memory.store(obs, action, reward, next_obs, done, vpred, ac_log)

        obs = next_obs
        episode_reward += reward
        # wandb.log({"rews": episode_reward})


        if done:
            break

    # After collecting enough data, compute loss and update the network
        optimizer.zero_grad()
        batch_size = 16  # Adjust based on your requirement
        if len(memory) >= batch_size:
            loss = compute_loss(memory)
            loss.backward()
            optimizer.step()
            memory.clear()

