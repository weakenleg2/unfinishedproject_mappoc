# noqa
"""
# Simple Spread

```{figure} mpe_simple_spread.gif
:width: 140px
:name: simple_spread
```

This environment is part of the <a href='..'>MPE environments</a>. Please read that
 page first for general information.

| Import               | `from pettingzoo.mpe import simple_spread_v2` |
|----------------------|-----------------------------------------------|
| Actions              | Discrete/Continuous                           |
| Parallel API         | Yes                                           |
| Manual Control       | No                                            |
| Agents               | `agents= [agent_0, agent_1, agent_2]`         |
| Agents               | 3                                             |
| Action Shape         | (5)                                           |
| Action Values        | Discrete(5)/Box(0.0, 1.0, (5))                |
| Observation Shape    | (18)                                          |
| Observation Values   | (-inf,inf)                                    |
| State Shape          | (54,)                                         |
| State Values         | (-inf,inf)                                    |
State: Represents the complete and precise description of the environment at a 
given point in time. It encapsulates all the information needed to predict how 
the environment will respond to any given action in that state. the state might be 
a concatenation of all agents' observations, velocities, positions, and 
communication information, as well as landmark positions.

Observation: Is what an agent can perceive from the state, and it might 
be a partial or noisy view of the state. In partially observable environments 
(like most real-world scenarios), agents cannot see the full state, and they need 
to act based on these limited observations.From the observation function, 
it appears the observation for each agent includes:
The agent's own position and velocity (agent.state.p_pos and agent.state.p_vel).
Positions of all landmarks (entity_pos).
Communication information or messages from other agents (comm).

This environment has N agents, N landmarks (default N=3). At a high level, agents
 must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent 
is to each landmark (sum of the minimum distances). Locally, the agents are penalized 
if they collide with other agents (-1 for each collision). The relative weights of 
these rewards can be controlled with the
`local_ratio` parameter.

Agent observations: `[self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions
, communication]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
there's a parameter max_cycles, which is set to a default value of 25. This parameter 
determines the maximum number of steps or frames for each episode in the environment. 
Once this number of steps is reached, the episode will terminate, and the environment 
will need to be reset to start a new episode.
```



`N`:  number of agents and landmarks

`local_ratio`:  Weight applied to local reward and global reward. Global reward weight 
will always be 1 - local reward weight.

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

"""

import numpy as np
from gymnasium.utils import EzPickle

# In the context of reinforcement learning and environments, this 
# could be used to save the state of an environment to disk and then 
# load it again later. This is particularly useful in scenarios where 
# you want to save the environment's state during a training process and 
# resume from that exact state later.
from pettingzoo.utils.conversions import parallel_wrapper_fn
# In multi-agent environments, there are often different modes or ways 
# to interact with the environment. One such mode is the parallel mode. 
# In this mode, instead of agents taking turns to act (like in a sequential setup), 
# all agents take actions simultaneously. This mode is quite similar to the traditional 
# single-agent Gym environment setup.onverts a PettingZoo environment 
# into a parallel environment. When you apply this wrapper to an environment, 
# you can interact with it using the parallel API. This means that at each step, 
# you would provide the actions for all agents at once and receive observations, 
# rewards, etc., for all agents simultaneously.

from custom_envs.mpe.core import Agent, Landmark, World
from custom_envs.mpe.scenario import BaseScenario
from custom_envs.mpe.simple_env import SimpleEnv, make_env
# simple_spread_c.py

# This file likely contains the specific logic for the "Simple Spread" scenario, 
# such as defining rewards, observations, and agent interactions for this specific 
# problem. This is where the environment's "rules" are defined.
# core.py

# This file seems to contain core components and utilities that are reused across 
# different scenarios. It likely defines classes or functions for things like agents, 
# entities, and worlds, which are then extended or used in specific scenarios.
# simple_env.py

# This file appears to wrap everything into a Gym-compatible environment. It uses 
# PettingZoo's AECEnv to provide an API that can be used in reinforcement learning 
# frameworks. This file contains the main environment class (SimpleEnv) that manages 
# the world state, agent interactions, and rendering.
# Workflow:

#     Initialization: When you create an instance of SimpleEnv, it takes parameters 
# like scenario, world, etc., to initialize the environment. It likely uses the scenario 
# object (from simple_spread_c.py) to set initial world state and agent properties.

#     Action and Observation Spaces: SimpleEnv defines action and observation spaces,
#  which tells you what kind of actions agents can take and what kind of observations 
# they will receive.

#     Stepping through the Environment: The step function in SimpleEnv progresses the 
# environment by one timestep. This function likely uses scenario-specific logic (
# from simple_spread_c.py) to update the world state and calculate rewards.

#     Rendering: The render function provides a way to visualize the environment. This 
# is useful for debugging and understanding agent behavior.

#     Utility Functions: Functions like observe, reset, and seed are also provided 
# for standard Gym environment functionality.

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        penalty_ratio = 0.5,
        full_comm=True,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,#####
    ):
        EzPickle.__init__(
            self, N=N, penalty_ratio=penalty_ratio,  
            local_ratio=local_ratio, full_comm=full_comm,
            max_cycles=max_cycles, continuous_actions=continuous_actions, 
            render_mode=render_mode
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N, penalty_ratio, full_comm)
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_v2"
        #         这一行调用的是 raw_env 的超类的构造函数 (__init__)。由于 raw_env 
        # 同时继承自 SimpleEnv 和 EzPickle，Python 的方法解析顺序决定了 SimpleEnv 的构
        # 造函数将被 super() 调用（只要 SimpleEnv 没有继承自 EzPickle）。

        #           使用 super() 可以确保父类（SimpleEnv）所需的任何设置都会被执行。
        # 这在面向对象编程中尤为重要，因为当你使用继承类时，你需要确保父类的属性在子类中添加或覆盖
        # 之前已正确设置。
        #  the name of the environment is being set as "simple_spread_v2". 
        # This kind of metadata can be useful in various contexts, such as 
        # when you want to retrieve or display information about the environment 
        # without needing to probe its internal state deeply.

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def action_callback(self, agent, _): 
      #To test full comm
      if self.full_comm:
        agent.action.c = np.array([1, 0])
        #Agents seem to have a binary communication channel represented by 
        # agent.action.c. Based on this, agents can either send a message or not.

      if agent.action.c[0] > agent.action.c[1]:
        self.last_message[agent.name] = np.concatenate((agent.state.p_pos, [0]))
        # only communicate position
        agent.color = np.array([0, 1, 0])
      else:
        agent.color = np.array([0.35, 0.35, 0.85])
        self.last_message[agent.name][-1] += 1
      
      return agent.action
    #     The agent's position (agent.state.p_pos) is concatenated
    #  with a [0] and stored in the self.last_message dictionary under 
    # the key of the agent's name. This seems to represent a message that 
    # includes the agent's position and a 'counter' or 'state' initialized to 0.
    #     The agent's color is set to green ([0, 1, 0]).

    # If the first value is not greater than the second, then:

    #     The agent's color is set to a shade of blue ([0.35, 0.35, 0.85]).
    #     The last value in the self.last_message for that agent is incremented by 1. 
    # This seems to be a counter, incrementing every time the agent doesn't send the 
    # main message.

    def make_world(self, N=3, penalty_ratio=0.5, full_comm=False):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = N
        num_landmarks = N
        self.n_collisions = 0
        world.collaborative = True
        self.full_comm = full_comm
        self.penalty_ratio = penalty_ratio 
        self.last_message = {}
        self.world_min = -1 - (0.1 * num_agents)
        self.world_max = 1 + (0.1 * num_agents)

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            # The purpose of using enumerate() here is to obtain 
            # both the index and the agent object without having to
            #  manually manage the index
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size = 0.15
            agent.action_callback = self.action_callback
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
            # # communication channel dimensionality
            # self.dim_c = 0
            # # position dimensionality
            # self.dim_p = 2
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world):
        # random properties for agents
        self.n_collisions = 0
        for _, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # random properties for landmarks
        for _, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(self.world_min, self.world_max, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for _, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(self.world_min, self.world_max, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)
    # This function is used to calculate various metrics (or benchmark data) for 
    # a given agent in a specific world setup. The returned metrics include the 
    # cumulative reward for the agent, the number of collisions involving the agent, 
    # the cumulative minimum distances from the landmarks, and the number of landmarks 
    # occupied by agents.

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    def reward(self, agent, world, global_reward=None):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent.collide:
            for a in world.agents:
                if a.name == agent.name:
                    continue
                is_collision = (self.is_collision(a, agent))
                if is_collision:
                    self.n_collisions += 1
                rew -= 1.0 * is_collision

        #Add penalty for communication
        if global_reward and agent.action.c[0] > agent.action.c[1]:
          rew += global_reward * self.penalty_ratio
        return rew

    def global_reward(self, world):
        # dis
        rew = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew
        # the global reward is indeed negative and is calculated based 
        # on the distances of the agents to the landmarks in the world.

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos)
        # communication of all other agents
        entity_pos = np.concatenate(entity_pos)
        comm = []
        for other in world.agents:
            #com_flag = 0
            if other is agent:
                continue

            message = self.last_message[other.name]
            #else:
                #message[-1] += 1

            #self.last_message[other.name] = message
            
            #if other.action.c is not None and other.action.c[0] > other.action.c[1]:
                #self.last_message[other.name] 
                #com_flag = 1

            comm.append(message)

        comm = np.concatenate(comm)
        obs = np.concatenate(
            (agent.state.p_pos, agent.state.p_vel,
            entity_pos, comm))

        return obs
    # This observation function constructs an observation for the agent in its 
    # current environment. The observation contains information about the agent
    # 's position and velocity, the positions of landmarks, and messages from 
    # other agents.

# Let's break down the function step by step:

#     Landmark Positions:
#         Initialize an empty list entity_pos to hold the positions of landmarks.
#         Loop through each landmark in the world and append its position 
# (entity.state.p_pos) to the entity_pos list.
#         Use np.concatenate to flatten entity_pos into a 1D array.

#     Communication from Other Agents:
#         Initialize an empty list comm to hold the communication messages 
# from other agents.
#         Loop through each agent in the world (other):
#             If the current agent (other) is the same as the agent for 
# whom we're constructing the observation, skip it (i.e., agents do not receive 
# their own messages).
#             Get the last message sent by other from self.last_message[other.name] 
# and append it to the comm list.

#     Note: There are commented lines of code that might suggest previous or 
# alternative methods of handling communication. You can ignore these for now 
# unless you plan to modify or understand previous implementations.
#         Use np.concatenate to flatten comm into a 1D array.

#     Construct the Observation:
#         Concatenate the agent's position (agent.state.p_pos), velocity 
# (agent.state.p_vel), flattened landmark positions (entity_pos), and flattened 
# communication messages (comm) to form the final observation.

#     Return:
#         Return the constructed observation (obs).

# So, the output observation is essentially a 1D array containing:

#     The agent's current position and velocity.
#     The positions of all landmarks.
#     The last messages sent by all other agents.

# This observation provides the agent with a comprehensive view of 
# its environment at any given time step, which allows the agent to make 
# informed decisions based on its surroundings and the messages received 
# from other agents.