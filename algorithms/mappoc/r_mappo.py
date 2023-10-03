import numpy as np
import torch
import torch.nn as nn
from algorithms.mappo.utils.util import get_gard_norm, huber_loss, mse_loss
#Eucliden norm,三种损失
from algorithms.mappo.utils.valuenorm import ValueNorm
#normalization of input 
from algorithms.mappo.algorithms.utils.util import check
#if input is numpy, convert to torch, if not, keep it

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update. actor network?
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        # ppo clip parameter (default: 0.2)
        # clip(r(theta),1-clip_para,1+clip_para)
        self.ppo_epoch = args.ppo_epoch
        # number of ppo epochs 15, for example, 10 PPO epochs, it means 
        # you'll be updating the policy 10 times using the same collected 
        # data before you go back and collect new data from the environment.
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        #  refers to the sequence length or the number of timesteps 
        # that are propagated through the recurrent layers during a
        #  forward and backward pass. This is often referred to as the 
        # "rollout length" or "sequence length." 10
        # In reinforcement learning, a "rollout" refers to the process of 
        # generating a sequence of states, actions, and rewards by following a
        #  policy (or a specific action-selection method) from an initial state
        #  to a terminal state rollout 和 trajectory差不多就是一个概念
        self.value_loss_coef = args.value_loss_coef
        # 0.5
        self.entropy_coef = args.entropy_coef
        # 0.01
        self.max_grad_norm = args.max_grad_norm    
        # max norm of gradients 0.5   
        self.huber_delta = args.huber_delta
        # coefficient of huber loss.  

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        # by default, use huber loss
        self._use_popart = args.use_popart
        # by default True, use PopArt to normalize rewards.
        # PopArt aims to adaptively normalize the targets (i.e., the returns) 
        # used in training the value function, while preserving the original 
        # outputs of the network for policy improvement. Here's a brief overview
        #  of how PopArt works:

        # Adaptive Normalization: Maintain a running estimate of the mean and
        #  variance of the returns. When updating the value function, normalize
        #  the target returns using this estimated mean and variance.

        # Preserving Outputs: When the normalization statistics (mean and variance) 
        # are updated, the weights and biases of the network are adjusted such that 
        # the outputs of the network (before the new normalization) remain the same 
        # as they were before the update. This ensures that the policy, which relies 
        # on the outputs of the value function, doesn't see abrupt changes.

        # Denormalization: When using the outputs of the network for policy decisions,
        #  the outputs are denormalized using the current normalization statistics.
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        # Incomplete Sequences: In some settings, especially with recurrent 
        # policies or value functions, if a sequence is not of a required length 
        # (e.g., shorter than a specified rollout length), it might be padded. 
        # The padding data can be considered "useless" for the value function update.

        # Off-Policy Data: In some algorithms, data collected under an old policy
        #  might be considered less relevant or "useless" for updating the value
        #  function under the current policy.

        # Specific Sampling Methods: Depending on how experiences are sampled 
        # (e.g., prioritized replay), some experiences might be given more importance 
        # than others
        self._use_policy_active_masks = args.use_policy_active_masks
        # whether to mask useless data in policy loss
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        # 两种不同的norm方法
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        # h 表示多智能体系统中每个智能体的参与者 RNN 的初始隐藏状态。
        # the hidden state 
        # h captures the temporal dependencies or memory of past observations and actions,
        #  allowing the agent to make decisions based on a sequence of past experiences.
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        # value function clip
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            # true
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
            # This step multiplies the value loss by the active masks. active_masks_batch 
            # is a tensor where each element corresponds to an agent's activity status
            #  (typically 1 if the agent is active and 0 if it's inactive or "dead"). 
            # By multiplying value_loss with active_masks_batch, you're effectively 
            # zeroing out the loss for inactive agents and keeping the loss for the 
            # active agents.
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample
        # unpack，后续再查

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        # torch.float32
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # log(\pi)-log(\pi_old),adv_targ 就是advantage estimate

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # surr1 和 surr2相当于求L^{CLIP}

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        # The negative sign (-) is present because in optimization, typically 
        # one minimizes the loss. However, the PPO objective is something we 
        # aim to maximize.
        # The dim=-1 argument in the torch.sum() function refers to summing
        #  over the last dimension of the tensor.
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
        #  This is the entropy of the action distribution produced by the policy. 
        # Entropy is a measure of randomness or unpredictability in the distribution.
        #  Maximizing entropy can help in promoting exploration in reinforcement learning,
        # The term - dist_entropy * self.entropy_coef is added to the policy loss to encourage 
        # exploration. Here's how:

        # If dist_entropy is high (meaning the action distribution is quite random), the subtracted
        #  value will be larger, making the overall objective lower, which is better as we 
        # are trying to minimize this combined objective.
        # If dist_entropy is low (meaning the action distribution is deterministic), the subtracted
        #  value is smaller, making the combined objective higher.
        # self.entropy_coef is a hyperparameter that weights how much emphasis we put on maximizing
        #  entropy. If it's set to 0, then entropy has no effect on the overall objective.
        #  If it's set to a positive value, it encourages exploration.

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
    # Training (train):

# The central training loop.
# First, it computes the advantages, which are essential for the PPO update. 
# The advantages represent how much better an action was compared to the expected 
# value from the critic.
# It then loops through the PPO epochs and gets data batches using data generators.
# For each batch, it performs a PPO update.
# The training information (like losses, gradient norms, etc.) is aggregated over
#  all updates.
# Preparation Methods (prep_training and prep_rollout):

# prep_training sets the policy's actor and critic to training mode.
# prep_rollout sets the policy's actor and critic to evaluation mode.
