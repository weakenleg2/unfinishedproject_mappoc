import torch
import numpy as np
import time
import copy
from torch.optim import Adam
import psutil
import copy
import sys
from algorithms.mappoc.dataset import Dataset
from algorithms.mappoc import logger
import time
from collections import deque
from mpi4py import MPI
from gym import spaces
import os
import shutil
from torch.distributions import Normal, kl_divergence
import torch
import torch.nn.functional as F

import wandb
wandb.init(project="mappoc_multiwalker_fixed", entity="2017920898")


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]




def lineup(func, count):
    x = []
    for i in range(count):
        x.append(copy.copy(func))
    return x



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1] # False = 0, True = 1, nonterminal if next step is not new
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]



def noise(deviation, x, ending):
    factor = deviation * (ending-iters_so_far)/ending
    if factor <= 0.0:
        factor = 0.0
    noise = np.random.normal(0.0, factor)
    x += noise
    # print(f"x:{x}")
    x = clip(x, -1.0, 1.0)
    return x

def renoise(deviation, x, begin, end):
    # looks like a tent
    # only on from begin
    if iters_so_far >= begin:
        # catch cases where we are above end
        if iters_so_far >= end:
            renfactor = 0.0
        # within tent do
        else:
            # first half
            if iters_so_far <= (begin+((end-begin)/2)):
                renfactor = deviation*2*((iters_so_far-begin)/(end-begin))
            # second half:
            else:
                renfactor = deviation*(1-((2*iters_so_far-(end+begin))/(end-begin)))
    else:
        renfactor = 0.0
    # add noise and return x
    noises = np.random.normal(0.0, renfactor)
    x += noises
    x = clip(x, -1.0, 1.0)
    #print(factor, 'renoise')
    return x

def clip(value, low, high):
    if value >= high:
        value = high
    elif value <= low:
        value = low
    return value

def are_networks_equal(net1, net2):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        if param1.data.ne(param2.data).sum() > 0:
            print("false")
            return False
    print("true")
    return True

def compute_losses(acs, atargs, rets, lrmults, ops, term_adv, pol_ov_op_ents, agents, oldagents, clip_param, entcoeff):
    # losses = []
    # print(f"acs:{acs.shape}")
    # for i in range(len(agents)):
        # Probability ratio for actions
    acs_tensor = torch.tensor(acs, dtype=torch.float32)

    logp_new = agents.pd.log_prob(acs_tensor).sum(dim=-1)
    logp_old = oldagents.pd.log_prob(acs_tensor).sum(dim=-1)
    meankl = torch.mean(kl_divergence(agents.pd,oldagents.pd))
    # 维度存疑
    ratio = torch.exp(logp_new - logp_old)

    # Clipped surrogate objective
    # print(f"ratio:{ratio}")
    # print(f"ratio:{type(ratio)}")
    # print(f"atargs:{atargs.shape}")
    surr1 =  (ratio.detach().numpy() * atargs)
    surr2 = torch.clamp(ratio, 1.0 - clip_param * lrmults, 1.0 + clip_param * lrmults).detach().numpy() * atargs
    pol_surr = -torch.min(torch.tensor(surr1), torch.tensor(surr2)).mean()
    # print(pol_surr)

    # Value function loss
    # print(f"agents.vpred:{type(agents.vpred)}")
    # print(f"rets:{type(rets)}")
    vf_loss = F.mse_loss(agents.vpred, torch.tensor(rets))
    # print(vf_loss.shape)

    # Entropy
    entropy = agents.pd.entropy().mean()
    pol_entpen = - entcoeff * entropy

    # Calculate logarithm of probability of policy over options

    log_pi = torch.log(torch.clamp(agents.op_pi, min=1e-5, max=1.0))
    # print(log_pi)
    old_log_pi = torch.log(torch.clamp(oldagents.op_pi, min=1e-5, max=1.0))
    
    # Calculate entropy of policy over options
    entropies = -(agents.op_pi * log_pi).sum(dim=1)
    
    # Calculate the PPO update for the policy over options
    # print(ops[0])
    ratio_pol_ov_op = torch.exp(log_pi.permute(1, 0)[ops[0]] - old_log_pi.permute(1, 0)[ops[0]])
    # print(ratio_pol_ov_op)
    
    # Surrogate from conservative policy iteration
    term_adv_clip = term_adv
    # print(term_adv_clip)
    surr1_pol_ov_op = ratio_pol_ov_op.detach().numpy() * term_adv_clip
    surr2_pol_ov_op = torch.clamp(ratio_pol_ov_op, 1.0 - clip_param, 1.0 + clip_param).detach().numpy() * term_adv_clip
    pol_surr_pol_ov_op = -torch.min(torch.tensor(surr1_pol_ov_op), torch.tensor(surr2_pol_ov_op)).mean()
    
    # Calculate the option policy loss
    # print(pol_surr_pol_ov_op)
    # print(pol_ov_op_ents * entropies.sum())
    op_loss = pol_surr_pol_ov_op - pol_ov_op_ents * entropies.sum()

    # Total loss
    # print("shape")
    # print(pol_surr.shape)
    # print(vf_loss.shape)
    # print(op_loss.shape)
    total_loss = pol_surr + vf_loss + op_loss

    # You might want to add other parts like option policy losses, etc.
    # Append the calculated losses to the losses list
    losses= [pol_surr, pol_entpen, vf_loss, meankl, entropy, op_loss]

    return total_loss, losses


def update_old_policy(old_model, current_model):
    for old_param, current_param in zip(old_model.parameters(), current_model.parameters()):
        if old_param.data.shape != current_param.data.shape:
            print(f"Mismatch! Old param shape: {old_param.data.shape}, current param shape: {current_param.data.shape}")
            raise ValueError("Parameter shapes do not match")
        old_param.data.copy_(current_param.data)




def traj_segment_generator_torch(agents, env, horizon, num, comm_weight, explo_iters, begintime, stoptime,
                           deviation, stochastic, num_options, seed):
    # Initialization and setup code...
    # ...
    begin = begintime
    stop = stoptime
    start=time.time()
    t = 0
    comm_weight = comm_weight
    timer = 0.1
    glob_count = 0
    glob_count_thresh = -1
    states = lineup([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]+(num)*copy.copy([0,0,0,0]), num)
    # 17
    options = lineup(1, num) # Causing initial communication
    acts = lineup([1,1,1,1], num)
    obs = []
    for i in range(num):
        # num of agents
        # adding the package informations and stats
        obs.append(states[i][:3] + states[i][:17])
        # 20, 前三维保存旧包？
        # adding the communication and timers
        for j in range(num):
            if j == i:
                continue
            else:
                obs[i] += states[i][17 + j * 4:17 + (j + 1) * 4] + [0]
                #  specifically a sub-list that starts at the index 17 + j * 4 
                # and ends just before the index 17 + (j + 1) * 4, 添加了5个元素
        # adding the real distances
        for j in range(num):
            if j == i:
                continue
            else:
                obs[i] += states[i][17 + j * 4:17 + (j + 1) * 4]
        # 29维
    
    rews = lineup([0], num)
    news = lineup(True, num)
    term = lineup(True, num)
    term_p = lineup([], num)
    vpreds = lineup([], num)
    logstds = lineup([], num)
    ep_rets = lineup([], num)
    ep_lens = lineup([], num)
    cur_ep_ret = lineup(0, num)
    cur_ep_len = lineup(0, num)
    curr_opt_duration = lineup(0, num)
    print('...Done')
    # Initialize history lists
    print('Initializing history arrays...')
    optpol_p_hist = lineup([], num)
    value_val_hist = lineup([], num)
    obs_hist = []
    rews_hist = []
    realrews_hist = []
    vpreds_hist = []
    news_hist = []
    opts_hist = []
    acts_hist = []
    opt_duration_hist = []
    logstds_hist = []

    # 像是insert
    for agent in env.agents:
        obs_hist.append([obs[i] for _ in range(horizon)])
        rews_hist.append(np.zeros(horizon, 'float32'))
        realrews_hist.append(np.zeros(horizon, 'float32'))
        vpreds_hist.append(np.zeros(horizon, 'float32'))
        news_hist.append(np.zeros(horizon, 'int32'))
        opts_hist.append(np.zeros(horizon, 'int32'))
        acts_hist.append([acts[i] for _ in range(horizon)])
        opt_duration_hist.append([[] for _ in range(num_options)])
        logstds_hist.append([[] for _ in range(num_options)])
    print("...Done")        
    print('Performing initial step...')
    # perform initial action
    action = [1, 1, 1, 1]
    for i in range(num):
        env.step(action)
    # Gather initial information
    for i in range(num):
        states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]], \
                                        env.terminations[env.agents[i]]
    # save correct obs
    for i in range(num):
        obs[i][0:3] = states[i][0:3] # Saving package_old, only in initial step
        obs[i][3:20] = states[i][:17] # Overwriting package_new and stats
    # perform initial communication应该只有前20是他自己的state，其他应该是别的agents的
    # 下面这段是采取初始action之后，再次更新
    for i in range(num):
        # Receive communication
        for j in range(num):
            if j == i:
                continue
            # 自身不通信
            else:
                if j <= i:
                    k = j
                else:
                    k = j-1
            # 保证更新当前代理？states来更新？
            if options[j] == 1:
                obs[i][20 + (k * 5): 20 + (k+1) * 5] = np.concatenate([states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
            else:
                # Timer goes up
                obs[i][20 + 4 + k * 5] += timer
        # Save the actual distances
        for j in range(num):
            if j == i:
                continue
            else:
                if j <= i:
                    k = j
                else:
                    k = j-1
            obs[i][20+(num-1)*5+k*4 : 20+(num-1)*5+(k+1)*4]=states[i][17+j*4:17+(j+1)*4]
    ###


    while True:

        ### Apply action实际上还是有这个交互过程，产生了，action，q_value,logprob
        for i in range(num):
            # Receive actions and vpreds
            # agent靠传递
            # print(f"obs{i}:{len(obs[i])}")
            acts[i], vpreds[i], _, logstds[i] = agents[i].act(stochastic, obs[i], options[i])
            # print(f"acts1:{logstds[i]}")
            # acts[i], vpreds[i], _, logstds[i] = agents[i].act(stochastic, obs[i], options[i])
            # print(f"acts2:{logstds[i]}")

            #  = np.random.uniform(-1, 1, 4)

            # 网络的东西了
            
            if t > 0 and t % horizon == 0:
                print(f"acts[i]:{acts[i]}")
            logstds_hist[i][options[i]].append(copy.copy(logstds[i]))
            # add noise for encouraging exploration
            for k in range(4):
                acts[i][k] = noise(deviation, copy.copy(acts[i][k]), explo_iters)
            for k in range(4):
                acts[i][k] = renoise(deviation, copy.copy(acts[i][k]), begin, stop)
        ###

        ### Check if horizon is reached and yield segs
        if t > 0 and t % horizon == 0:

            # Update the missing ep_rets
            for i in range(num):
                print(f"acts[i]:{acts[i]}")
                ep_rets[i].append(cur_ep_ret[i])
                cur_ep_ret[i] = 0
                ep_lens[i].append(cur_ep_len[i])
                cur_ep_len[i]=0
            # Create new segs
            segs = []
            for i in range(num):
                segs.append({"ob": np.array(obs_hist[i]), "rew": np.array(rews_hist[i]), "realrew": realrews_hist[i],
                             "vpred" : np.array(vpreds_hist[i]), "new" : news_hist[i], "ac" : np.array(acts_hist[i]), "opts" : opts_hist[i],
                             "nextvpred": vpreds[i] * (1 - news[i]), "ep_rets" : ep_rets[i],"ep_lens" : ep_lens[i],
                             'term_p': term_p[i], 'value_val': value_val_hist[i], "opt_dur": opt_duration_hist[i],
                             "optpol_p": optpol_p_hist[i],"logstds": logstds_hist[i]})
            print('...Done, duration: ', str(int(time.time() - start)), 'seconds')
            yield segs
            # Restart process
            start=time.time()
            ep_rets = lineup([], num)
            ep_lens = lineup([], num)
            term_p = lineup([], num)
            value_val_hist = lineup([], num)
            opt_duration_hist = lineup([[] for _ in range(num_options)], num)
            logstds_hist = lineup([[] for _ in range(num_options)], num)
            curr_opt_duration = lineup(0., num)
            # 这边应该是到达边界后，进行数据储存并且refresh
        ###

        ### Save generated data
        j = t % horizon
        for i in range(num):
            obs_hist[i][j] = copy.copy(obs[i])
            vpreds_hist[i][j] = copy.copy(vpreds[i])
            news_hist[i][j] = copy.copy(news[i])
            acts_hist[i][j] = copy.copy(acts[i])
            opts_hist[i][j] = copy.copy(options[i])
        ###

        ### Apply step function
        # Apply the gathered actions to the environment
        for i in range(num):
            if np.isnan(np.min(acts[i])) == True:
                acts[i] = [0, 0, 0, 0]
                print('WARNING: NAN DETECTED')
            env.step(copy.copy(acts[i])) # Careful: Without this "copy" operation acts is modified
        # Receive the new states, rews and news
        for i in range(num):
            states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]],\
                                          env.terminations[env.agents[i]]
        # Update package_old in case there was communication in this step
        for i in range(num):
            if options[i] == 1:
                obs[i][0:3] = copy.copy(states[i][0:3])
        # combine reward and do logging
        for i in range(num):
            rews[i] = copy.copy(rews[i])*1.0
            if options[i] == 1:
                rews[i] = rews[i] - comm_weight
            # 惩罚
            rews[i] = rews[i]/10 if num_options >1 else rews[i] # To stabilize learning.
            cur_ep_ret[i] += rews[i]*10 if num_options > 1 else rews[i]
            cur_ep_len[i] += 1
            rews_hist[i][j] = rews[i]
            realrews_hist[i][j] = rews[i]
            curr_opt_duration[i] += 1
        ###

        ### Update the observation with new values
        # save correct obs
        for i in range(num):
            obs[i][3:20] = copy.copy(states[i][:17])  # Updating package_new and stats
        # Calculate option
        for i in range(num):
            options[i] = agents[i].get_option(copy.copy(obs[i])) 
            
            # Might be unnecessary since no manipulation happens
            opt_duration_hist[i][options[i]].append(curr_opt_duration[i])
            curr_opt_duration[i] = 0.
        # Receive communication
        for i in range(num):
            for j in range(num):
                if j == i:
                    continue
                else:
                    if j <= i:
                        k = j
                    else:
                        k = j - 1
                if options[j] == 1:
                    obs[i][20 + (k * 5): 20 + (k + 1) * 5] = np.concatenate(
                        [states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
                else:
                    obs[i][20 + 4 + k * 5] += timer # Timer goes up
            # Save the actual distances for vpred
            for j in range(num):
                if j == i:
                    continue
                else:
                    if j <= i:
                        k = j
                    else:
                        k = j - 1
                obs[i][20 + (num - 1) * 5 + k * 4: 20 + (num - 1) * 5 + (k + 1) * 4] = states[i][17 + j * 4:17 + (j + 1) * 4]
        # Only for logging
        for i in range(num):
            t_p = []
            v_val = []
            for oopt in range(num_options):
                # print(f"obs[i]:{len(obs[i])}")
                # 此处他index两次，但是不现实,trajectory应该只改了这边
                v_val.append(agents[i].get_vpred(obs[i],oopt)[0])
                t_p.append(agents[i].get_tpred(obs[i],oopt))
            term_p[i].append(t_p)
            optpol_p_hist[i].append(agents[i]._get_op(obs[i]))
            value_val_hist[i].append(v_val)
            # print(f"agents[i].termination_sample:{agents[i].termination_sample[0]}")
            term[i] = agents[i].termination_sample[0]
        ###

        # Check if termination of episode happens
        if any(news):
            # if new rollout starts -> reset last action and start anew
            for i in range(num):
                news[i] = True
                ep_rets[i].append(cur_ep_ret[i])
                cur_ep_ret[i] = 0
                ep_lens[i].append(cur_ep_len[i])
                cur_ep_len[i] = 0
            env.reset(seed=seed)
            # perform initial action
            for i in range(num):
                env.step(action)
            # Gather initial information
            for i in range(num):
                states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]], \
                                              env.terminations[env.agents[i]]
                options[i] = 1
            # save correct obs
            for i in range(num):
                obs[i][3:20] = states[i][:17]  # Overwriting package_new and stats
                obs[i][0:3] = states[i][0:3]  # Saving package_old, only in initial step
            # Perform initial communication
            for i in range(num):
                # Receive communication
                for j in range(num):
                    if j == i:
                        continue
                    else:
                        if j <= i:
                            k = j
                        else:
                            k = j - 1
                    if options[j] == 1:
                        obs[i][20 + (k * 5): 20 + (k + 1) * 5] = np.concatenate([states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
                    else:  # Just a relic, never going to happen
                        # Timer goes up
                        obs[i][20 + 4 + k * 5] += 1
                # Save the actual distances
                for j in range(num):
                    if j == i:
                        continue
                    else:
                        if j <= i:
                            k = j
                        else:
                            k = j - 1
                    obs[i][20 + (num - 1) * 5 + k * 4: 20 + (num - 1) * 5 + (k + 1) * 4] = states[i][17 + j * 4:17 + (j + 1) * 4]
            ###
        t += 1
# let us back to this function again, it is used to interact with the env and 
# generate corresponding data right? let me tell you my understanding now, you 
# can add, if I miss something, first it initializes some data, using list, 
# then take action [1,1,1,1] especially to initialize the shape of obs and how
#  to update obs. Then we enter the while true loop to generate data, then we 
#   use networks to get action, prob, q_value, then we check if it is horizon,
# if it is, we save and yield data, if not, we step to get new obs, reward and 
# termination, then update obs as before, last we consider what if we meet termination.
def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
        #   有了
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          comm_weight, explo_iters, begintime, stoptime, deviation, # factors for traj generation
          entro_iters, final_entro_iters, pol_ov_op_ent, final_pol_ov_op_ent, # entropy parameters
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          num_options=1,
          app='',
          saves=False,
          wsaves=False,
          epoch=-1,
          seed=1,
          dc=0,
          num,
          device):

    ### Fundamental definitions
    evals = False
    optim_batchsize_ideal = optim_batchsize
    np.random.seed(seed)
    torch.manual_seed(seed)    
    
   

    ### Setup all tensorflow functions and dependencies
    # Setup all functions
    ac_space = spaces.Box(low=np.array([-1,-1,-1,-1]),high=np.array([1,1,1,1]), shape=(4,),dtype=np.float32)
    # add the dimension in the observation space
    q_space_shape = (17+(num-1)*(5+4),)
    pi_space_shape = (17+(num-1)*(5),)
    mu_space_shape = (6,)
    ac_space_shape = (4,)
    # 这个也先不管obs shape
    # create lists to store agents with policies and variables in
    agents = []
    oldagents = []
    lrmults = []
    clip_params = []
    atargs = []
    rets = []
    pol_ov_op_ents = []
    obs = []
    ops = []
    term_adv = [] ## NO NAME CHANGE
    acs = []
    rews = []
    # kloldnew = [] ## NO NAME CHANGE
    # ents = []
    # meankl = [] ## NO NAME CHANGE
    # meanents = []
    # pol_entpen = [] ## NO NAME CHANGE
    # ratios = []
    # atarg_clips = []
    # surr1 = []
    # surr2 = []
    # pol_surr = []
    # vf_losses = []
    # total_losses = []
    losses = []
    # log_pi = [] ## NO NAME CHANGE
    # old_log_pi = []
    # entropies = [] ## NOT ENTS
    # ratio_pol_ov_op = [] ## NO NAME CHANGE
    # term_adv_clip = []
    # surr1_pol_ov_op = [] ## NO NAME CHANGE
    # surr2_pol_ov_op = [] ## NO NAME CHANGE
    # pol_surr_pol_ov_op = [] ## NO NAME CHANGE
    # op_loss = []
    for i in range(num):
        agents.append(policy_func(q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Construct network for new policy
        # oldagents.append(policy_func( q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Network for old policy
        obs.append([])
        ops.append([])
        acs.append([])
        atargs.append([])
        rews.append([])
    # print(f"agents:{agents}")
    # for i in range(num):
    #     for old_param, current_param in zip(oldagents[i].parameters(), agents[i].parameters()):
    #         # if old_param.data.shape != current_param.data.shape:
    #         print(f"Mismatch! Old param shape: {old_param.data.shape}, current param shape: {current_param.data.shape}")
            # print(f"old_model.parameters(), current_model.parameters():{oldagents[i].parameters().shape, agents[i].parameters().shape}")
        
   
    
    var_lists = []
    # term_lists = []
    # lossandgrads = []
    # adams = []
    # assign_old_eq_new = []
    # compute_losses = []
    
    optimizers = []

    for i in range(num):
    # 1. Getting Trainable Variables: Directly use PyTorch model.parameters() method
        var_lists.append(agents[i].parameters())
        optimizer = torch.optim.Adam(agents[i].parameters(),lr=1e-5,eps=adam_epsilon) 
        optimizers.append(optimizer)
        # assign_old_eq_new.append(update_old_policy(oldagents[i], agents[i]))
        # assign_old_eq_new.append(lambda i=i: update_old_policy(oldagents[i], agents[i]))


    
    ###

    
    
    
    oldagents = copy.deepcopy(agents)
    ###

    ### Start training process
    episodes_so_far = 0 # Can't be retrieved if using the epoch argument
    timesteps_so_far = 0
    if epoch >= 0:
        timesteps_so_far += timesteps_per_batch * epoch
    global iters_so_far
    iters_so_far = 0
    if epoch >= 0:
        iters_so_far += int(epoch)
    des_pol_op_ent = pol_ov_op_ent # define policy over options entropy scheduling
    # 传进来的，entropy para都是穿进来的
    if epoch>entro_iters:
        if epoch>=final_entro_iters:
            des_pol_op_ent=final_pol_ov_op_ent
        else:
            des_pol_op_ent=pol_ov_op_ent+(final_pol_ov_op_ent-pol_ov_op_ent)/(final_entro_iters-entro_iters)*(iters_so_far-entro_iters)
        des_pol_op_ent=des_pol_op_ent/final_pol_ov_op_ent # warning, possible conflict with exact iter of schedule
    max_val = -100000 # define max_val, this will be updated to always store the best model
    tstart = time.time()
    allrew = max_val-1
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    ########
    
    # for i in range(num):
    #     for old_param, current_param in zip(oldagents[i].parameters(), agents[i].parameters()):
    #         if old_param.data.shape != current_param.data.shape:
    #             print("Wrong")
    seg_gen = traj_segment_generator_torch(agents=agents, env=env, horizon=timesteps_per_batch, 
                                     num=num, comm_weight=comm_weight, 
                                     explo_iters=explo_iters, begintime=begintime,
                                     stoptime=stoptime, deviation=deviation, 
                                     stochastic=True, num_options=num_options, seed=seed)
    seg_gen_old = traj_segment_generator_torch(agents=oldagents, env=env, horizon=timesteps_per_batch, 
                                     num=num, comm_weight=comm_weight, 
                                     explo_iters=explo_iters, begintime=begintime,
                                     stoptime=stoptime, deviation=deviation, 
                                     stochastic=True, num_options=num_options, seed=seed)
    # oldagents = copy.deepcopy(agents)

    # for i in range(num):
    #     for old_param, current_param in zip(oldagents[i].parameters(), agents[i].parameters()):
    #         # if old_param.data.shape != current_param.data.shape:
    #         print(f"Mismatch! Old param shape: {old_param.data.shape}, current param shape: {current_param.data.shape}")
    # “ob”：观察结果
    # “rew”：奖励
    # “realrew”：真实奖励（可能在某些标准化或缩放之前）
    # “vpred”：价值预测
    # “new”：指示环境是否重置的标志（剧集结束）
    # “ac”：代理采取的操作
    # “opts”：代理使用的选项或模式（如果是基于选项的策略）
    # “nextvpred”：用于价值函数估计中引导的下一个值预测
    # “ep_rets”：episode return
    # “ep_lens”：episode length
    # “term_p”：终止概率（如果适用，对于基于选项的策略）
    # “value_val”：价值函数评估
    # “opt_dur”：选项持续时间
    # “optpol_p”：选项策略概率
    # “logstds”：标准差的对数（对于随机策略有用）
    # 字典中的每个键对应于智能体与环境交互的不同方面，相关联的值是在轨迹段期间收集的数据
    ##########
    segs = []
    datas = []
    savbuffer = lineup(deque(maxlen=100), num)
    comm_savings = lineup(0, num)
    lenbuffer = lineup(deque(maxlen=100), num)
    rewbuffer = lineup(deque(maxlen=100), num)
    realrew = lineup([], num)
    for i in range(num):
        datas.append([0 for _ in range(num_options)])
        segs.append([])
    ###

    ### Start training loop
    while True:
        losssaver=lineup([[], []], num)
        # Some error collecting
        if callback: callback(locals(), globals())
        

        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

                
        if schedule == 'constant':
            cur_lrmult = 1.0
            # 初始是这个值
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
    
                
        # adapt the entropy to the desired value
        if (iters_so_far+1)>entro_iters:
            if iters_so_far>final_entro_iters:
                des_pol_op_ent=final_pol_ov_op_ent
            else:
                des_pol_op_ent=pol_ov_op_ent+(final_pol_ov_op_ent-pol_ov_op_ent)/(final_entro_iters-entro_iters)*(iters_so_far-entro_iters)
        
        


        ### Starting the training iteration
        logger.log("*********** Iteration %i *************" % iters_so_far)
        # Sample (s,a)-Transitions
        print('Sampling trajectories...')
        # for i in range(num):
        #     print(f"Agent {i} parameter shapes before trajectory generation:")
        #     for param in agents[i].parameters():
        #         print(param.data.shape)
        # for i in range(num):
        #     for old_param, current_param in zip(oldagents[i].parameters(), agents[i].parameters()):
        #         if old_param.data.shape != current_param.data.shape:
        #             print("Wrong")
        #         if old_param.data.shape == current_param.data.shape:
        #             print("Right")
        segs = seg_gen.__next__()
        # oldagents = copy.deepcopy(agents)

        seg_gen_olds = seg_gen_old.__next__()

        # for i in range(num):
        #     print(f"Agent {i} parameter shapes after trajectory generation:")
        #     for param in agents[i].parameters():
        #         print(param.data.shape)
        # Evaluation sequence for one iteration
        # for i in range(num):
        #     for old_param, current_param in zip(oldagents[i].parameters(), agents[i].parameters()):
        #         if old_param.data.shape != current_param.data.shape:
        #             print("Wrong")
        #         if old_param.data.shape == current_param.data.shape:
        #             print("Right")
        segment=[]
        if wsaves:
            print("Optimizing...")
            start_opt = time.time()
        # splitting by agent i
        lrlocal = lineup([],num)
        listoflrpairs = lineup([],num)
        
        for i in range(num):
            #print('-----------Agent', str(i + 1), '------------')
            add_vtarg_and_adv(segs[i], gamma, lam) # Calculate A(s,a,o) using GAE
            
            
            
            # calculate information for logging
            opt_d = []
            std = []
            for j in range(num_options):
                dur = np.mean(segs[i]['opt_dur'][j]) if len(segs[i]['opt_dur'][j]) > 0 else 0.
                opt_d.append(dur)
                logstd = np.mean(segs[i]['logstds'][j]) if len(segs[i]['logstds'][j]) > 0 else 0.
                std.append(np.exp(logstd))
            print("mean std of agent ", str(i+1), ":", std)
            
            obs[i], acs[i], ops[i], atargs[i],rews[i], tdlamret = segs[i]["ob"], segs[i]["ac"], segs[i]["opts"], segs[i]["adv"],segs[i]["rew"],\
                                                          segs[i]["tdlamret"]
            #vpredbefore = segs[i]["vpred"] # predicted value function before udpate
            atargs[i] = (atargs[i] - atargs[i].mean()) / atargs[i].std() # standardized advantage function estimate
            if hasattr(agents[i], "ob_rms"): agents[i].ob_rms.update(obs[i]) # update running mean/std for policy
            if hasattr(agents[i], "ob_rms_only"): agents[i].ob_rms_only.update(obs[i])
            oldagents[i].load_state_dict(agents[i].state_dict())

            
            # minimum batch size:
            min_batch=160
            t_advs = [[] for _ in range(num_options)]
            # select all the samples concerning one of the options
            for opt in range(num_options):
                indices = np.where(ops[i]==opt)[0]
                #print("batch size:",indices.size, "for opt ", opt)
                opt_d[opt] = indices.size
                if not indices.size:
                    t_advs[opt].append(0.)
                    continue
                # This part is only necessary when we use options. We proceed to these verifications in order not to discard any collected trajectories.
                if datas[i][opt] != 0:
                    if (indices.size < min_batch and datas[i][opt].n > min_batch):
                        datas[i][opt] = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices], atarg=atargs[i][indices]
                                                     ,rews = rews[i][indices],
                                                     vtarg=tdlamret[indices]), shuffle=not agents[i].recurrent)
                        t_advs[opt].append(0.)
                        continue
                    elif indices.size + datas[i][opt].n < min_batch:
                        # pdb.set_trace()
                        oldmap = datas[i][opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],obs[i][indices]))
                        cat_ac = np.concatenate((oldmap['ac'],acs[i][indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atargs[i][indices]))
                        cat_rews = np.concatenate((oldmap['rews'],rews[i][indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[i][opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, rews = cat_rews,vtarg=cat_vtarg),
                                                shuffle=not agents[i].recurrent)
                        t_advs[opt].append(0.)
                        continue
                    elif (indices.size + datas[i][opt].n > min_batch and datas[i][opt].n < min_batch)\
                            or (indices.size > min_batch and datas[i][opt].n < min_batch):
                        oldmap = datas[i][opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],obs[i][indices]))
                        cat_ac = np.concatenate((oldmap['ac'],acs[i][indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atargs[i][indices]))
                        cat_rews = np.concatenate((oldmap['rews'],rews[i][indices]))

                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[i][opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, rews = cat_rews,vtarg=cat_vtarg),
                                                    shuffle=not agents[i].recurrent)
                    if (indices.size > min_batch and datas[i][opt].n > min_batch):
                        datas[i][opt] = d = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices],
                                                         atarg=atargs[i][indices], rews = rews[i][indices], vtarg=tdlamret[indices]),
                                                    shuffle=not agents[i].recurrent)
                elif datas[i][opt] == 0:
                    datas[i][opt] = d = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices], atarg=atargs[i][indices],rews = rews[i][indices],
                                                     vtarg=tdlamret[indices]), shuffle=not agents[i].recurrent)
                    # print(len(obs[i][indices]))
                # define the batchsize of the optimizer:
                
                optim_batchsize = optim_batchsize or obs[i].shape[0]
                # print(f"optim_batchsize:{obs[i].shape[0]}")
                #print("optim epochs:", optim_epochs)
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    # k = 0
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        # k+=1
                        # print(d.iterate_once(optim_batchsize))
                        # print(optim_batchsize)

                        # Calculate advantage for using specific option here
                        # for i in range(optim_batchsize):
                        tadv,nodc_adv = agents[i].get_opt_adv(batch["ob"],opt)
                        tadv = tadv if num_options > 1 else np.zeros_like(tadv)
                        t_advs[opt].append(nodc_adv)
                        
                        
                        optimizers[i].zero_grad()

    
                        # Extracting data from the batch
                        # observations = batch["ob"]
                        # observations = torch.FloatTensor(observations).to(device)
                        actions = batch["ac"]
                        advantages = batch["atarg"]
                        rets = batch["rews"]

                        # ... (you might have more data to extract like rewards, options, etc.)
                        
                        # Forward pass: Make predictions or compute the output using the model
                        # agents.append(policy_func( q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Construct network for new policy
                        # oldagents.append(policy_func( q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Network for old policy
                        # Creating a learnable parameter in PyTorch
                        
                        # Calculate the loss between the predictions and the target values
                        # print(f"len(cur_lrmult):{cur_lrmult}")
                        loss,loss_append = compute_losses( actions, advantages, rets, cur_lrmult, [opt], 
                                              tadv, des_pol_op_ent, agents[i], oldagents[i], clip_param, entcoeff)
                        # wandb.log({"loss": loss})

                        #  ops, 
                        # print("#################")
                        # print("done")
                        # Backward pass: Compute the gradient of the loss w.r.t the model parameters
                        loss.backward()
                        
                        # Perform a single optimization step (parameter update)
                        # for optimizer in optimizers:
                        optimizers[i].step()
                        # print("#################")
                        # print("done")
                        # print(loss_append)
                        losses.append(loss_append)

                    # for i in range(num):
                    # are_networks_equal(agents[i],oldagents[i])
                    # are_networks_equal(agents[i],oldagents[i])

                    # 检查parameter等不等
                    # print(k)
                    # oldagents = copy.deepcopy(agents)


                    # at the end of batch, save all the mean losses
                    # print("len")
                    # print(losses)
                    # print(len(losses))
                    # shape不太一样
                    # print(type(losses))
                    # print(losses[0])
                    # print(len(losses[0]))
                    # losses = torch.stack(losses)
                    losses_detached = []
                    for row in losses:
                        row_detached = [tensor.detach().numpy() for tensor in row]
                        losses_detached.append(row_detached)
                    tempsaver=[0,0,0,0,0,0]
                    for n in range(len(losses[0])):                
                        # losses_detached = [loss.detach().numpy() for loss in losses]
                        k=n+1
                        tempsaver[n]=np.mean(np.array(losses_detached)[:,n:k])
                    losssaver[i][opt].append(tempsaver.copy())
            ###

            # do logging:
            lrlocal[i] = copy.copy((segs[i]["ep_lens"], segs[i]["ep_rets"]))  # local values
            listoflrpairs[i] = (MPI.COMM_WORLD.allgather(lrlocal[i]))  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs[i]))

            lrlocal[i] = copy.copy((segs[i]["ep_lens"], segs[i]["ep_rets"])) # local values
            listoflrpairs[i] = (MPI.COMM_WORLD.allgather(lrlocal[i])) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs[i]))
            lenbuffer[i].extend(lens)
            rewbuffer[i].extend(rews)
            for k in range(num):
                realrew[i]=np.mean(rews)
            # wandb.log({"rcomm_savings": comm_savings}, step=step)

            comm_savings[i] = len(np.where(ops[i]==0)[0])/timesteps_per_batch
            # wandb.log({"comm_savings": comm_savings})
            savbuffer[i].extend([comm_savings[i]])
            # The last started episode pulls down the average and is therefore discarded in statistics
            # logger.record_tabular("EpLenMean of Agent " + str(i+1) + ": ", np.mean(lenbuffer[i][:,:-1]))
            # logger.record_tabular("EpRewMean of Agent " + str(i + 1) + ": ", np.mean(rewbuffer[i]))
            # logger.record_tabular("Comm_sav of Agent " + str(i + 1) + ": ", comm_savings[i])

           
       
        if wsaves:
            print('...Done, duration: ', str(int(time.time() - start_opt)), 'seconds')
        ### Group Book keeping
        allrew=0
        sumrew=0
        for i in range(num):
            sumrew += np.mean(rewbuffer[i])
        allrew = sumrew/num
        realrealrew = np.mean(realrew)
        wandb.log({"rews": realrealrew})


        avg_comm_save = 0
        for i in range(num):
            avg_comm_save += comm_savings[i]
        avg_comm_save = avg_comm_save/num
        avg_group_comm_save = 0
        for i in range(num):
            avg_group_comm_save += np.mean(savbuffer[i])
        avg_group_comm_save = avg_group_comm_save/num
        wandb.log({"comm_savings": avg_group_comm_save})

        

        ### Final logging
        iters_so_far += 1
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        ###
