import mdp as MDP
import RLCore
import random
import numpy as np

#  Calculates a matrix that stores the transition probabilities
# We store it so it needs not be calculated often
def getTransitionMatrix(mdp: MDP.mdp) -> 'list[list[list[float]]]':
    transitionF = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    return [[[transitionF(s1, a, s2) for s2 in mdp.states] for a in mdp.actions] for s1 in mdp.states]

def policyIteration(mdp):
    policy = {s : random.choice(mdp.actions) for s in mdp.states}
    V = {s : 0 for s in mdp.states}
    transition = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    changes = 1
    iterations = 0
    while changes:
        iterations += 1
        changes = 0
        # evaluate
        P = [transition(s1, policy[s1], s2) for s2 in mdp.states for s1 in mdp.states]
        P = np.array(P, float).reshape((len(mdp.states), len(mdp.states)))
        P = P * mdp.discount
        matinv = np.linalg.inv(np.identity(len(mdp.states)) - P)
        V = np.dot([mdp.reward(s, policy[s]) for s in mdp.states], matinv)
        # improve
        for s in mdp.states:
            # maximize a such that r_a(s) + discount*P_a(s)*V(s)
            def actionValue(a):
                ret = mdp.reward(s, a) + (mdp.discount * np.dot([transition(s,a,s2) for s2 in mdp.states],V))
                return ret
            newBest = max(mdp.actions, key=actionValue)
            if (policy[s] != newBest):
                changes += 1
                policy[s] = newBest
    print(f"Iterations: {iterations}")
    return policy, V

if __name__ == '__main__':
    import RiverSwim
    from ValueIteration import valueIteration

    rs = RiverSwim.riverswim(8, 0.55, 0.05, 0.1, 1.0, 0.9)
    
    print("Value-iteration")
    Q = valueIteration(rs)
    RiverSwim.printV(Q)
    RiverSwim.printActions(Q)

    print("Policy-iteration")
    p, v = policyIteration(rs)
    print(p)
    print(np.round(v,2))
