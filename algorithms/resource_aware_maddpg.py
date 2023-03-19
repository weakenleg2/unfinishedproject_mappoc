import torch
from torch import Tensor
from torch.autograd import Variable
import copy
from utils.neuralnets import MLPNetwork
from utils.noise import OUNoise
from utils.misc import soft_update, gumbel_softmax

MSELoss = torch.nn.MSELoss()

class RA_MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, in_dim, out_dim, ctrl_critic_in, opt_critic_in, n_agents=3, constrain_out=True, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
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
        self.control_policy = MLPNetwork(in_dim, out_dim, hidden_dim=hidden_dim, 
                                         discrete_action=discrete_action, constrain_out=constrain_out).to(device)
        self.options_policy = MLPNetwork(in_dim, 2, hidden_dim=hidden_dim,
                                         discrete_action=True).to(device)

        self.control_critic = MLPNetwork(ctrl_critic_in, 1, hidden_dim=hidden_dim,
                                         discrete_action=False).to(device)
        self.options_critic = MLPNetwork(opt_critic_in, 1, hidden_dim=hidden_dim,
                                         discrete_action=False).to(device)

        self.target_control_policy = copy.deepcopy(self.control_policy)
        self.target_options_policy = copy.deepcopy(self.options_policy)
        self.target_control_critic = copy.deepcopy(self.control_critic)
        self.target_options_critic = copy.deepcopy(self.options_critic)

        self.control_policy_optimizer = torch.optim.Adam(self.control_policy.parameters(), lr=lr)
        self.options_policy_optimizer = torch.optim.Adam(self.options_policy.parameters(), lr=lr)
        self.control_critic_optimizer = torch.optim.Adam(self.control_critic.parameters(), lr=lr)
        self.options_critic_optimizer = torch.optim.Adam(self.options_critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.exploration = OUNoise(out_dim)

        self.init_dict = {"lr": lr, 
                          "in_dim": in_dim,
                          "out_dim": out_dim,
                          "ctrl_critic_in": ctrl_critic_in,
                          "opt_critic_in": opt_critic_in,
                          "n_agents": n_agents,
                          "discrete_action": discrete_action,
                          "gamma": gamma, "tau": tau,}
    
    def _get_actions(self, obs):
      control = self.control_policy(obs)
      comm = self.options_policy(obs)
      return torch.cat((control, comm), dim=-1)

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
      for obs in observations:
        action = self._get_actions(obs).to('cpu')

        if explore:
          cont = action[:2]
          cont = cont + Variable(Tensor(self.exploration.noise()),
                                  requires_grad=False)
          discrete = action[2:]
          discrete = gumbel_softmax(discrete.unsqueeze(0), hard=True).squeeze()
          action = torch.cat((cont, discrete), dim=0)

        action = action.clamp(-1, 1)
        actions.append(action.detach())

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
        ctrl = [acs[i][:, :2] for i in range(self.n_agents)]
        opt = [acs[i][:, 2:] for i in range(self.n_agents)]

        # Control critic loss
        self.control_critic_optimizer.zero_grad()
        self.options_critic_optimizer.zero_grad()

        all_trgt_acs = [self.control_policy(nobs) for nobs in next_obs]
        trgt_ctrl_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        target_ctrl_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        self.target_control_critic(trgt_ctrl_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        vf_in_ctrl = torch.cat((*obs, *ctrl), dim=1)
        vf_in_opt = torch.cat((*obs, *opt), dim=1)

        # Options critic loss
        all_trgt_opt= [self.options_policy(nobs) for nobs in next_obs]
        trgt_opt_vf_in = torch.cat((*next_obs, *all_trgt_opt), dim=1)
        target_opt_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        self.target_options_critic(trgt_opt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        actual_ctrl_value = self.control_critic(vf_in_ctrl)
        actual_opt_value = self.options_critic(vf_in_opt)

        vf_opt_loss = MSELoss(actual_opt_value, target_opt_value.detach())
        vf_ctrl_loss = MSELoss(actual_ctrl_value, target_ctrl_value.detach())

        vf_opt_loss.backward()
        vf_ctrl_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.control_critic.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.options_critic.parameters(), 0.5)

        self.control_critic_optimizer.step()
        self.options_critic_optimizer.step()

        self.control_policy_optimizer.zero_grad()
        self.options_policy_optimizer.zero_grad()

        curr_ctrl = self.control_policy(obs[agent_i])
        curr_opt = self.options_policy(obs[agent_i])

        all_ctrl = []
        all_opt = []
        for i, ob in zip(range(self.n_agents), obs):
            if i == agent_i:
                all_ctrl.append(curr_ctrl)
                all_opt.append(curr_opt)
            else:
                all_ctrl.append(self.control_policy(ob))
                all_opt.append(self.options_policy(ob))

        vf_ctrl_in = torch.cat((*obs, *all_ctrl), dim=1)
        vf_opt_in = torch.cat((*obs, *all_opt), dim=1)

        pol_ctrl_loss = -self.control_critic(vf_ctrl_in).mean()
        pol_opt_loss = -self.options_critic(vf_opt_in).mean()

        pol_ctrl_loss += (curr_ctrl**2).mean() * 1e-3
        pol_opt_loss += (curr_opt**2).mean() * 1e-3

        pol_ctrl_loss.backward()
        pol_opt_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.control_policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.options_policy.parameters(), 0.5)
        
        self.control_policy_optimizer.step()
        self.options_policy_optimizer.step()

        #if logger is not None:
            #logger.add_scalars('agent%i/losses' % agent_i,
                               #{'vf_loss': vf_loss,
                                #'pol_loss': pol_loss},
                               #self.niter)

    def to_device(self, device):
      self.control_policy.to(device)
      self.options_policy.to(device)
      self.control_critic.to(device)
      self.options_critic.to(device)
      self.target_control_critic.to(device)
      self.target_options_critic.to(device)     

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_control_critic, self.control_critic, self.tau)
        soft_update(self.target_options_critic, self.options_critic, self.tau)

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # self.prep_training(
        # device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                    "control_policy": self.control_policy.state_dict(),
                    "options_policy": self.options_policy.state_dict(),
                    "control_critic": self.control_critic.state_dict(),
                    "options_critic": self.options_critic.state_dict(),

                    "control_optimizer": self.control_policy_optimizer.state_dict(),
                    "options_optimizer": self.options_policy_optimizer.state_dict(),
                    "control_critic_optimizer": self.control_critic_optimizer.state_dict(),
                    "options_critic_optimizer": self.options_critic_optimizer.state_dict(),
                     }
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
        instance.control_policy.load_state_dict = save_dict["control_policy"]
        instance.options_policy.load_state_dict = save_dict["options_policy"]
        instance.control_critic.load_state_dict = save_dict["control_critic"]
        instance.options_critic.load_state_dict = save_dict["options_critic"]

        instance.control_policy_optimizer.load_state_dict = save_dict["control_optimizer"]
        instance.options_policy_optimizer.load_state_dict = save_dict["options_optimizer"]
        instance.control_critic_optimizer.load_state_dict = save_dict["control_critic_optimizer"]
        instance.options_critic_optimizer.load_state_dict = save_dict["options_critic_optimizer"]

        instance.to_device(device)

        return instance
