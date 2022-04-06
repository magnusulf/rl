from typing import Any, Callable, TypeVar
import numpy as np
import RLCore
import MDP

S = TypeVar('S')
A = TypeVar('A')

# Given a starting state and an action it gives the discounted, expected value
# Of performing the action
def getActionValue(discount, mdp, stateFrom, action, P, R, Q):
    def prob(stateTo): return P[mdp.stateIdx(stateFrom)][mdp.actionIdx(action)][mdp.stateIdx(stateTo)]
    def value(stateTo): return RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, R, discount, stateFrom, action, stateTo, Q)
    def probValue(stateTo): return value(stateTo) * prob(stateTo)
    return sum([probValue(s) for s in mdp.states])


def valueIteration(mdp: MDP.mdp) -> 'list[list[float]]':
    P = MDP.getTransitionMatrix(mdp)
    R = MDP.getRewardMatrix(mdp)
    Q = [[0.0 for _ in mdp.actions] for _ in mdp.states]
    changes: int = 1
    iterations: int = 0
    while (changes != 0):
        iterations += 1
        changes = 0
        for state in mdp.states:
            for action in mdp.actions:
                newVal = getActionValue(mdp.discount, mdp, state, action, P, R, Q)
                oldVal = Q[mdp.stateIdx(state)][mdp.actionIdx(action)]
                if (newVal != oldVal):
                    changes = changes + 1
                Q[mdp.stateIdx(state)][mdp.actionIdx(action)] = newVal
    print("Iterations: " + str(iterations))
    return Q
    