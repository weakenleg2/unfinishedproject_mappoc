import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import copy
from utils.neuralnets import MLPNetwork
from utils.noise import OUNoise
from utils.misc import soft_update, gumbel_softmax, onehot_from_logits

MSELoss = torch.nn.MSELoss()

class RA_MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, in_dim, out_dim, n_agents=3, 
                 eps=1.0, eps_decay=1e6, gamma=0.95, 
                 tau=0.01, lr=0.01, 
                 actor_hidden_dim=128,
                 critic_hidden_dim=128,
                 discrete_action=False, device='cpu'):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic

            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.n_agents = n_agents
        self.control_actions = out_dim - 2
        policy_out = (self.control_actions)
        critic_in = n_agents * (in_dim + out_dim)

        self.control_policies = []
        self.options_policies = []
        self.control_optimizers = []
        self.options_optimizers = []

        for _ in range(n_agents):
          control_policy = MLPNetwork(in_dim, policy_out, hidden_dim=actor_hidden_dim, 
                                         discrete_action=discrete_action, constrain_out=False).to(device)
          options_policy = MLPNetwork(in_dim, 2, hidden_dim=actor_hidden_dim,
                                         discrete_action=True).to(device)

          control_policy_optimizer = torch.optim.Adam(control_policy.parameters(), lr=lr)
          options_policy_optimizer = torch.optim.Adam(options_policy.parameters(), lr=lr)

          self.control_policies.append(control_policy)
          self.options_policies.append(options_policy)
          self.control_optimizers.append(control_policy_optimizer)
          self.options_optimizers.append(options_policy_optimizer)


        self.critic = MLPNetwork(critic_in, 1, hidden_dim=critic_hidden_dim,
                                         discrete_action=False, constrain_out=False).to(device)
        #self.target_control_policy = copy.deepcopy(self.control_policy)
        #self.target_options_policy = copy.deepcopy(self.options_policy)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.exploration = OUNoise(self.control_actions, scale=1)
        self.device = device
        self.curr_eps = eps 
        self.eps_decay = eps_decay
        self.n_iter = 0

        # Normalization
        self.obs_mean = np.zeros(in_dim)
        self.obs_M2= np.zeros(in_dim)
        self.k = 0

        self.init_dict = {"lr": lr, 
                          "in_dim": in_dim,
                          "out_dim": out_dim,
                          "eps_decay": eps_decay,
                          "n_agents": n_agents,
                          "discrete_action": discrete_action,
                          "gamma": gamma, "tau": tau,}
    
    def _get_actions(self, obs, agent):
      obs = self.normalize(obs)
      control = self.control_policies[agent](obs)
      #control = control_params[:2]
      #control = (torch.randn(self.control_actions, device=self.device, requires_grad=True) * control_params[..., -2:]) + control_params[..., :-2]
      #control = torch.normal(control_params[..., :-2], torch.abs(control_params[..., -2:]))
      comm = self.options_policies[agent](obs)
      comm = onehot_from_logits(comm)
      return torch.cat((control, comm), dim=-1)

    def _get_target_actions(self, obs):
      control_params = self.target_control_policy(obs)
      control = (torch.randn(self.control_actions, device=self.device, requires_grad=True) * control_params[..., -2:]) + control_params[..., :-2]
      #control = torch.normal(control_params[..., :-2], torch.abs(control_params[..., -2:]))
      comm = self.target_options_policy(obs)
      comm = onehot_from_logits(comm)
      return torch.cat((control, comm), dim=-1)

    def _norm_single(self, obs):
      obs = np.array(obs)
      self.k += 1
      delta = obs - self.obs_mean
      
      self.obs_mean = self.obs_mean + delta / self.k
      self.obs_M2 = self.obs_M2 + (delta * (obs - self.obs_mean))

      if self.k < 2:
        self.variance = np.ones_like(self.obs_M2)
      else:
        self.variance = self.obs_M2 / (self.k - 1)

      std = self.variance ** 0.5

      obs = (obs - self.obs_mean) / (std + 1e-8)
      obs = np.clip(obs, -10, 10)
      obs = torch.tensor(obs, dtype=torch.float32)
      return obs

    def normalize(self, obs):
      if len(obs.shape) == 1:
        return self._norm_single(obs)
      else:
        return torch.stack([self._norm_single(o) for o in obs])
      
    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
    def reset_noise(self):
      self.exploration.reset()

    def step(self, observations, explore=False):
      """
      Take a step forward in environment with all agents
      Inputs:
          observations: List of observations for each agent
          explore (boolean): Whether or not to add exploration noise
      Outputs:
          actions: List of actions for each agent
      """
      actions = []
      observations = observations.squeeze()
      for i, obs in enumerate(observations):
        action = self._get_actions(obs, i).to('cpu')
        cont = action[:2]
        discrete = action[2:]

        if explore:
          if torch.rand(1) <= self.curr_eps:
            cont = (torch.rand(cont.shape) * 2) - 1

          discrete = gumbel_softmax(discrete.unsqueeze(0), hard=True).squeeze()
        else:
          discrete = onehot_from_logits(discrete)

        action = torch.cat((cont, discrete), dim=0)
        action = action.clamp(-1, 1)
        actions.append(action.detach())
      
      if self.curr_eps > 0.01:
        self.curr_eps -= (1 / self.eps_decay)

      return actions

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample

        # Critic loss

        all_trgt_acs = [self._get_actions(nobs, agent_id) for nobs, agent_id in zip(next_obs, range(self.n_agents))]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        self.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = self.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())

        self.critic_optimizer.zero_grad()
        vf_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        self.control_optimizers[agent_i].zero_grad()
        self.options_optimizers[agent_i].zero_grad()

        curr_acs= self._get_actions(obs[agent_i], agent_i)

        all_acs = []
        for i, ob in zip(range(self.n_agents), obs):
            if i == agent_i:
                all_acs.append(curr_acs)
            else:
                all_acs.append(self._get_actions(ob, i).detach())

        vf_in = torch.cat((*obs, *all_acs), dim=1)

        pol_loss = -self.critic(vf_in).mean()
        pol_loss += (curr_acs**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.control_policies[agent_i].parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.options_policies[agent_i].parameters(), 0.5)
        
        self.control_optimizers[agent_i].step()
        self.options_optimizers[agent_i].step()

        if logger is not None:
            logger.add_scalars('agent/losses',
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.n_iter)

    def update_all_targets(self, logger=None):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        #soft_update(self.target_control_policy, self.control_policy, self.tau)
        #soft_update(self.target_options_policy, self.options_policy, self.tau)

        soft_update(self.target_critic, self.critic, self.tau)
        self.n_iter += 1

        if logger:
          logger.add_scalar('agent/epsilon', self.curr_eps, self.n_iter)
          logger.add_scalar('other/mean', self.obs_mean.mean(), self.n_iter)
          logger.add_scalar('other/variance', self.variance.mean(), self.n_iter)

    def to_device(self, device):
      self.device = device
      self.critic.to(device)

      for i in range(self.n_agents):
        self.control_policies[i].to(device)
        self.options_policies[i].to(device)

      #self.target_control_policy.to(device)
      #self.target_options_policy.to(device)
      self.target_critic.to(device)



    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(
        # device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                    "critic": self.critic.state_dict(),
                    "n_iter": self.n_iter,
                    "curr_eps": self.curr_eps,

                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    }
        for i in range(self.n_agents):
            save_dict["control_policy_{}".format(i)] = self.control_policies[i].state_dict()
            save_dict["options_policy_{}".format(i)] = self.options_policies[i].state_dict()
            save_dict["control_optimizer_{}".format(i)] = self.control_optimizers[i].state_dict()
            save_dict["options_optimizer_{}".format(i)] = self.options_optimizers[i].state_dict()
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename, device='cuda'):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        device = torch.device(device)
        save_dict = torch.load(filename, map_location=device)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.critic.load_state_dict = save_dict["critic"]

        #load all the policies and optimizers
        for i in range(instance.n_agents):
          instance.control_policies[i].load_state_dict = save_dict["control_policy_{}".format(i)]
          instance.options_policies[i].load_state_dict = save_dict["options_policy_{}".format(i)]
          instance.control_optimizers[i].load_state_dict = save_dict["control_optimizer_{}".format(i)]
          instance.options_optimizers[i].load_state_dict = save_dict["options_optimizer_{}".format(i)]

        instance.critic_optimizer.load_state_dict = save_dict["critic_optimizer"]
        instance.device = device

        instance.n_iter = save_dict["n_iter"]
        instance.curr_eps = save_dict["curr_eps"]
        instance.to_device(device)

        return instance
