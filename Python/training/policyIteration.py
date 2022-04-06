from typing import Any
import MDP
import RLCore
import random
import numpy as np

#  Calculates a matrix that stores the transition probabilities
# We store it so it needs not be calculated often
def getTransitionMatrix(mdp: MDP.mdp) -> 'list[list[list[float]]]':
    transitionF = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    return [[[transitionF(s1, a, s2) for s2 in mdp.states] for a in mdp.actions] for s1 in mdp.states]

def policyIteration(mdp: MDP.mdp):

    policy: 'list[Any]' = [random.choice(mdp.actions) for _ in mdp.states]
    V: 'list[float]' = [0 for _ in mdp.states]
    transition = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    changes = 1
    iterations = 0
    while changes:
        iterations += 1
        changes = 0
        # evaluate
        P = [transition(s1, policy[mdp.stateIdx(s1)], s2) for s2 in mdp.states for s1 in mdp.states]
        P = np.array(P, float).reshape((len(mdp.states), len(mdp.states)))
        P = P * mdp.discount
        matinv = np.linalg.inv(np.identity(len(mdp.states)) - P)
        V = np.dot([mdp.reward(s, policy[mdp.stateIdx(s)]) for s in mdp.states], matinv)
        # improve
        for s in mdp.states:
            # maximize a such that r_a(s) + discount*P_a(s)*V(s)
            def actionValue(a):
                ret = mdp.reward(s, a) + (mdp.discount * np.dot([transition(s,a,s2) for s2 in mdp.states],V))
                return ret
            newBest = max(mdp.actions, key=actionValue)
            if (policy[mdp.stateIdx(s)] != newBest):
                changes += 1
                policy[mdp.stateIdx(s)] = newBest
    print(f"Iterations: {iterations}")
    return policy, V