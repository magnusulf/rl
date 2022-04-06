
from typing import Any, Callable, List, TypeVar
import MDP
import RLCore
import random as rnd
import numpy as np
import QLearning
import math

S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]


# Calculates a matrix that stores the rewards
# We store it so it needs not be calculated often
        

def ucbLearn(mdp: MDP.mdp[S, A], initialState: S, epsilon: float, delta: float) -> 'list[list[float]]' :
    policy = QLearning.policyEpsilonGreedy(mdp, 0.1)
    rewardMatrix = MDP.getRewardMatrix(mdp)
    getActionStateValue = lambda s1, a, s2, Q: RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, rewardMatrix, mdp.discount, s1, a, s2, Q)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)

    maxQ = 1.0/(1.0-mdp.discount)
    Q = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    Q_hat = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    c2 = 4* np.sqrt(2)/10

    epsilon2 = epsilon/3
    R = np.ceil(np.log(1/(epsilon2 * (1-mdp.discount)) / (1-mdp.discount)))
    L = np.floor(np.log2(R))
    xi_L = 1/(math.pow(2, L+2)) * epsilon2 * math.pow(np.log(1/(1-mdp.discount)),-1)
    M = max(np.ceil(2*np.log2(1/(xi_L * (1-mdp.discount)))),10)
    epsilon1 = epsilon /(24 * R * M * np.log(1/(1-mdp.discount)))
    H = np.log(1/((1-mdp.discount)*epsilon1))/np.log(1/mdp.discount)
    def iota(k: float): return np.log(len(mdp.states)*len(mdp.actions)*(k+1)*(k+2)/delta)
    def a(k: float): return (H+1)/(H+k)

    N = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState = initialState
    accReward = 0.0
    for i in range(5_000_000):
        action: A = policy(Q_hat, currentState)
        nextState = transitionF(currentState, action)
        reward = getActionStateValue(currentState, action, nextState, Q_hat)
        transitionReward = rewardMatrix[mdp.stateIdx(currentState)][mdp.actionIdx(action)]
        N[mdp.stateIdx(currentState)][mdp.actionIdx(action)] += 1
        
        k = N[mdp.stateIdx(currentState)][mdp.actionIdx(action)]
        bk = c2/(1-mdp.discount) * np.sqrt(H*iota(k)/k)
        learningRate = a(k)
        Q[mdp.stateIdx(currentState)][mdp.actionIdx(action)] = (1-learningRate) *  Q[mdp.stateIdx(currentState)][mdp.actionIdx(action)] + learningRate *(reward + bk)
        Q_hat[mdp.stateIdx(currentState)][mdp.actionIdx(action)] = min(Q_hat[mdp.stateIdx(currentState)][mdp.actionIdx(action)], Q[mdp.stateIdx(currentState)][mdp.actionIdx(action)])
        #if(i % 1000 == 0): print(bk)
        currentState = nextState
        accReward += transitionReward
    print(N)
    print("Accumulated reward: %.1f" % accReward)
    return Q_hat
