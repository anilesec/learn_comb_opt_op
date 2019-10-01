#!/usr/bin/env python
# coding: utf-8

import math
import random
import argparse
import numpy as np
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from utils.dqn_utils import plot


# ### Deep Q-Network
class DQN(nn.Module):
    """ Note: Computes the Q-value for the state-action pair(should be done in parallel)
        1.DQN network selection criteria should be discussed(hyper-parameterization)
        2.Num of actions in OP case keeps changing, but in DQN framework this should be same from training point of view
            - This should be addressed either by masking or inculcating -ve reward
        3.Num of actions(out_features) will change for each instance because the sol set sizes are different
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.fc2 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.fc3 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.fc4 = nn.Linear(in_features=3*embed_dim, out_features=1)

    def forward(self, state_action):
        """
        :param state_action: is a list of 3 embeddings:-
                        - u_S(vector representing partial solution)
                        - u_V/S(vector representing nodes that are not part of partial solution set S)
                        - u_v(vector representing action taken)
        :return: Q-value for a single state-action pair(state represented using u_S and u_V/S,
        action represented using u_v)
        """
        t = F.relu(torch.cat((self.fc1(state_action[0]), self.fc2(state_action[1]), self.fc3(state_action[2])), dim=0))
        t = self.fc4(t)

        return t


# Experience class
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity  # maximum limit for number of experiences
        self.memory = []  # structure to store the experiences
        self.push_count = 0  # number of experiences added to memory

    # push the experience to memory
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # if memory exceeds capacity then push experiences to
            # beginning and overwrite the oldest experiences
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # sample batch_size number of experiences from memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return "ReplayMemory"


# Epsilon Greedy Strategy
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

    def __repr__(self):
        return "EpsilonGreedyStrategy"


# Reinforcement Learning Agent
class Agent:
    def __init__(self, strategy, num_actions, device):
        """
        :param strategy: Epsilon strategy class as input to decide the
        strategy (explore or exploit) for selecting action to create experience
        :param num_actions: Number of possible actions that an agent can take
        from the given state(in op case, it should be set of all nodes)
        :param device: gpu or cpu device
        """
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit using policy net

    def __repr__(self):
        return "ReinforcementLearningAgent"


# Environment manager(manages op environment)
class OpEnvManager:
    def __init__(self, device):
        self.device = device
        # self.env = gym.make('CartPole-v0').unwrapped  # set the environment(not required)
        self.current_state = None  # this maybe redundant, think about it later
        self.reset()  # set the env to a starting state, if above initialization is done, \
        # then this line may not be required
        self.done = False  # tracks whether or not action taken has ended the episode or not

    def step(self, actions, action, current_state, depot):
        """
        This method takes actions, action, current_state, depot and returns new_state, action_reward,
        episode_done(T orF) and diag_info(diagnostics info)
        To be done:
            1. new_state: obtained by appending action to current_state
            2. action_reward:
             sum(action_prize, reward(by checking length constraint), reward(by checking node repetition))
            3. episode_done: check if the next action is same as depot node, then terminate
        """
        new_state = current_state.apppend(actions[action])
        action_reward = self.compute_reward(selected_action=actions[action])
        if action_reward is None:
            raise NotImplementedError('reward computation is not implemented')
        if new_state is depot:
            episode_done = True

        diag_info = "No diagnostics info available"
        return new_state, action_reward, episode_done, diag_info

    def compute_reward(self, selected_action):
        total_reward = node_prize_reward(selected_action) 
                    #  node_rep_reward(selected_action) + \
                    #  length_constraint_reward(selected_action)

        return total_reward

    def reset(self, depot=None):
        """
        Initially, current_state is depot node(usually index 0).
        Indicates that agent is at the start of the episode.
        Returns an initial observation(state) from the environment.
        """
        self.current_state = depot

    def close(self):
        raise NotImplementedError('not yet implemented, may not be useful')

    def render(self, mode='human'):
        raise NotImplementedError('not yet implemented, may not be useful')

    def num_actions_available(self, sol_set):
        return len(sol_set)  # needs modification for batch(note: num of actions may vary across batch, need to address)

    def take_action(self, action):
        _, reward, self.done, _ = self.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        # need to be updated
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like( self.current_screen )
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def __repr__(self):
        return "OpEnvironmentManager"


# ### Tensor processing
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


# ### Q-Value Calculator
class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser()


    opts = parser.parse_args()

    # Set hyperparameters
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 100000
    lr = 0.001
    num_episodes = 1000

    # set gpu or cpu for device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize environment manager
    em = OpEnvManager(device)

    # strategy to select action to create RM(Replay Memory)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    # initialize agent
    agent = Agent(strategy, em.num_actions_available(), device)

    # initialize RM
    memory = ReplayMemory(memory_size)

    # initialize policy network
    policy_network = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

    # initialize target network(clone of policy network)
    target_network = DQN(em.get_screen_height(), em.get_screen_width()).to(device)

    # load policy network parameters to target_network
    target_network.load_state_dict(policy_network.state_dict())

    # use target_network only in 'eval' mode(no training for target network)
    target_network.eval()

    # initialize optimizer(Adam)
    optimizer = optim.Adam(params=policy_network.parameters(), lr=lr)

    episode_durations = []
    for episode in range(num_episodes):
        em.reset()  # update required
        state = em.get_state()  # update required

        for time_step in count():
            action = agent.select_action(state, policy_network)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences_batch = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences_batch)

                current_q_values = QValues.get_current(policy_network, states, actions)
                next_q_values = QValues.get_next(target_network, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if em.done:
                episode_durations.append(time_step)
                plot(episode_durations, 100)
                break

        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())

    em.close()