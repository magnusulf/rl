from typing import Any, Callable, List, TypeVar
import MDP
import RLCore
import random as rnd
import numpy as np
import torch
from torch import nn
from tensorflow.keras.optimizers import Adam

S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]

def policyRandom(actions: 'list[A]') -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        return actions[rnd.randint(0, len(actions)-1)]
    return pol

def policyEpsilonGreedy(mdp: MDP.mdp[S, A], epsilon: float) -> 'Callable[[list[list[float]], S], A]' :
    def pol(net: nn.module, s: S):#Q: 'list[list[float]]'):
        rand = rnd.random()
        if (rand < epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            #Qvalues = Q[mdp.stateIdx(s)]
            Qvalues = net.pred(mdp.stateIdx(s))
            idx = np.argmax(Qvalues)
            return mdp.actions[idx]
    return pol

def qLearn(mdp: MDP.mdp[S, A], policy: Policy[S, A], initialState: S) -> 'list[list[float]]' :
    R = MDP.getRewardMatrix(mdp)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    maxQ = 1.0/(1.0-mdp.discount)
    QNet = QNetwork(len(mdp.states), len(mdp.actions))
    TargetNet = QNetwork(len(mdp.states), len(mdp.actions))
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    N = 50 # copy weights of QNet into TargetNet after N steps
    visitCount = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState = initialState
    accReward = 0.0
    for i in range(1_000_000):
        a = policy(QNet, currentState)
        nextState = transitionF(currentState, a)
        transitionReward = R[mdp.stateIdx(currentState)][mdp.actionIdx(a)]
        visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + 1
        k = 10.0
        learningRate: float = k /(k + (visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)]))
        accReward = accReward + transitionReward
        pred = QNet.predict(currentState)[mdp.actionIdx(a)]
        with torch.no_grad():
            reward = transitionReward + mdp.discount * np.argmax(TargetNet.predict(nextState))
        target = (1.0-learningRate) * pred + learningRate * reward
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if (i % N == 0):
            TargetNet.load_state_dict(QNet.state_dict())
        currentState = nextState
    print("Accumulated reward: %.1f" % accReward)
    return QNet

class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.embedding = nn.Embedding(num_states, 10)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_actions),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear_relu_stack(x)
        return x