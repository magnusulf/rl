from typing import Any, Callable, List, TypeVar

import torch
import MDP
import RLCore
import DQNCore
import random as rnd
import numpy as np

# from typing import Any, Callable, Generic, TypeVar
# import math
# import random
# #import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# from itertools import count
# from PIL import Image

# #import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

# class QNetwork(nn.Module):
#     def __init__(self, num_states, num_actions):
#         super(QNetwork, self).__init__()
#         self.embedding = nn.Embedding(num_states, 10)
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(10, 50),
#             nn.ReLU(),
#             nn.Linear(50, 50),
#             nn.ReLU(),
#             nn.Linear(50, num_actions),
#         )

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.linear_relu_stack(x)
#         return x

# device = torch.device("cpu")

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# def experienceToTransition(state, action, next_state, reward):
#     state = torch.tensor([state], device=device)
#     action = torch.tensor([action], device=device)
#     next_state = torch.tensor([next_state], device=device)
#     reward = torch.tensor([reward], device=device)
#     return Transition(state, action, next_state, reward)

# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

# class TwinClassifier():
#     def __init__(self, num_states, num_actions, memory_capacity, batch_size):
#         self.num_states = num_states
#         self.num_actions = num_actions
#         # q networks
#         self.policy_net = QNetwork(num_states, num_actions).to(device)
#         self.target_net = QNetwork(num_states, num_actions).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
#         self.optimizer = optim.Adam(self.policy_net.parameters())
#         # replay memory
#         self.memory = ReplayMemory(memory_capacity)
#         self.batch_size = batch_size

#     def optimize_model(self, discount):
#         transitions = self.memory.sample(self.batch_size)
#         batch = Transition(*zip(*transitions))

#         state_batch = torch.cat(batch.state)
#         action_batch = torch.cat(batch.action)
#         next_state_batch = torch.cat(batch.next_state)
#         reward_batch = torch.cat(batch.reward)

#         state_action_values = self.policy_net(state_batch).gather(1, action_batch)
#         next_state_values = self.target_net(torch.cat(batch.next_state)).max(1)[0].detach()
#         expected_state_action_values = (next_state_values * discount) + reward_batch

#         criterion = nn.SmoothL1Loss()
#         loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         #for param in policy_net.parameters():
#         #    param.grad.data.clamp_(-1, 1)
#         self.optimizer.step()

#     def update_target_network(self):
#         self.target_net.load_state_dict(self.policy_net.state_dict())



S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]

def policyRandom(actions: 'list[A]') -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        return actions[rnd.randint(0, len(actions)-1)]
    return pol

def policyEpsilonGreedy(mdp: MDP.mdp[S, A], epsilon: float) -> 'Callable[[list[list[float]], S], A]' :
    def pol(policy_net, s: S, device):
        rand = rnd.random()
        if (rand < epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            Qvalues = policy_net(torch.tensor([mdp.stateIdx(s)], device=device)).detach().numpy()
            idx = np.argmax(Qvalues)
            return mdp.actions[idx]
    return pol
        

def dqLearn(mdp: MDP.mdp[S, A], policy: Policy[S, A], initialState: S) -> 'list[list[float]]' :
    R = MDP.getRewardMatrix(mdp)
    getActionStateValue = lambda s1, a, s2, Q: RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, R, mdp.discount, s1, a, s2, Q)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    Qnet = DQNCore.TwinClassifier(len(mdp.states), len(mdp.actions), 1_000, batch_size=1)
    #maxQ = 1.0/(1.0-mdp.discount)
    #Q = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    visitCount = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState = initialState
    accReward = 0.0
    for i in range(100):
        action = policy(Qnet.policy_net, currentState, DQNCore.device)
        nextState = transitionF(currentState, action)
        transitionReward = R[mdp.stateIdx(currentState)][mdp.actionIdx(action)]
        #reward = getActionStateValue(currentState, a, nextState, Q)
        transition = DQNCore.experienceToTransition(mdp.stateIdx(currentState), [mdp.actionIdx(action)], mdp.stateIdx(nextState), transitionReward)
        Qnet.memory.push(*transition)
        # visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + 1
        # k = 10.0
        # learningRate: float = k /(k + (visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)]))
        accReward = accReward + transitionReward

        #Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = (1.0-learningRate) * Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + learningRate * reward
        Qnet.optimize_model(mdp.discount)

        currentState = nextState
        if (i % 10 == 0):
            Qnet.update_target_network()
        if (i % 100 == 0):
            print(i)
    print("Accumulated reward: %.1f" % accReward)
    stateIdxs = [mdp.stateIdx(s) for s in mdp.states]
    return Qnet.policy_net(torch.tensor([stateIdxs], device=DQNCore.device))[0].detach().numpy()
