from typing import Any, Callable, TypeVar
import numpy as np
import RLCore
import mdprm

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')

# Given a starting state an action and the end state what is the value
# This is equal to the transition reward plus the discounted (expected) value of the end state
def getActionToStateValue(s1: S, u1: U, a: A, s2: S, u2: U, mdp: mdprm.mdprm, discount: float, R, Q: 'list[list[list[float]]]') -> float:
    transitionReward: float = R[mdp.stateIdx(s1)][mdp.actionIdx(a)][mdp.stateIdx(s2)][mdp.rewardStateIdx(u1)]
    nextStateValue: float = max(Q[mdp.stateIdx(s2)][mdp.rewardStateIdx(u2)])
    return transitionReward + discount * nextStateValue


# Given a starting state and an action it gives the discounted, expected value
# Of performing the action
def getActionValue(mdp: mdprm.mdprm[S, A, U], s1: S, u1: U, action: A, P, R, Q: 'list[list[list[float]]]') -> float:
    def prob(s2, u2): return P[mdp.stateIdx(s1)][mdp.rewardStateIdx(u1)][mdp.actionIdx(action)][mdp.stateIdx(s2)][mdp.rewardStateIdx(u2)]
    def value(s2, u2): return getActionToStateValue(s1, u1, action, s2, u2, mdp, mdp.discount, R, Q)
    def probValue(s2, u2):
        val = value(s2, u2) * prob(s2, u2)
        return val
    return sum([probValue(s, u) for s in mdp.states for u in mdp.reward_states])


def valueIteration(mdp: mdprm.mdprm) -> 'list[list[list[float]]]':
    P = mdprm.getTransitionMatrix(mdp)
    R = mdprm.getRewardMatrix(mdp)
    
    Q: 'list[list[list[float]]]' = [[[0.0 for _ in mdp.actions] for _ in mdp.reward_states] for _ in mdp.states]
    changes: int = 1
    iterations: int = 0
    while (changes != 0):
        iterations += 1
        changes = 0
        
        for u in mdp.reward_states:
            if (mdp.isTerminal(u)):
                continue
            for state in mdp.states:
                for action in mdp.actions:
                    newVal = getActionValue(mdp, state, u, action, P, R, Q)
                    oldVal = Q[mdp.stateIdx(state)][mdp.rewardStateIdx(u)][mdp.actionIdx(action)]
                    if (newVal != oldVal):
                        changes = changes + 1
                    Q[mdp.stateIdx(state)][mdp.rewardStateIdx(u)][mdp.actionIdx(action)] = newVal
        if (iterations > 80):
            break
    print("Iterations: " + str(iterations))
    return Q
    