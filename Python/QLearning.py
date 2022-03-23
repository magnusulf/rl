from typing import Any, Callable, List, TypeVar
import MDP
import RLCore
import random as rnd
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]


# Calculates a matrix that stores the rewards
# We store it so it needs not be calculated often
def getRewardMatrix(mdp: MDP.mdp) -> 'list[list[float]]':
    return [[mdp.reward(s, a) for a in mdp.actions] for s in mdp.states]

def policyRandom(actions: 'list[A]') -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        return actions[rnd.randint(0, len(actions)-1)]
    return pol

def policyEpsilonGreedy(mdp: MDP.mdp, epsilon: float) -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        rand = rnd.random()
        if (rand <= epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            Qvalues = Q[mdp.stateIdx(s)]
            if (sum(Qvalues) < 0.01): # Random when we barely know something
                return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
            else:
                idx = np.argmax(Qvalues)
                return mdp.actions[idx]
    return pol
        

def qLearn(mdp: MDP.mdp[S, A], policy: Policy[S, A], initialState: S) -> 'list[list[float]]' :
    R = getRewardMatrix(mdp)
    getActionStateValue = lambda s1, a, s2, Q: RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, R, mdp.discount, s1, a, s2, Q)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    maxQ = 1.0/(1.0-mdp.discount)
    Q = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    visitCount = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState = initialState
    accReward = 0.0
    for i in range(2_000_000):
        a = policy(Q, currentState)
        nextState = transitionF(currentState, a)
        reward = getActionStateValue(currentState, a, nextState, Q)
        transitionReward = R[mdp.stateIdx(currentState)][mdp.actionIdx(a)]
        visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + 1
        k = 10.0
        learningRate: float = k /(k + (visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)]))
        accReward = accReward + transitionReward
        Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = (1.0-learningRate) * Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + learningRate * reward
        currentState = nextState
    print("Accumulated reward: %.1f" % accReward)
    return Q
