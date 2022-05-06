from typing import Any, Callable, List, TypeVar
import MDP
import math
import RLCore
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]

def policyRandom(actions: 'list[A]') -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        return actions[rnd.randint(0, len(actions)-1)]
    return pol

def policyEpsilonGreedy(mdp: MDP.mdp[S, A], epsilon: float) -> 'Callable[[list[list[float]], S], A]' :
    def pol(Q: 'list[list[float]]', s: S):
        rand = rnd.random()
        if (rand < epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            Qvalues = Q[mdp.stateIdx(s)]
            idx = np.argmax(Qvalues)
            return mdp.actions[idx]
    return pol
        

def qLearn(mdp: MDP.mdp[S, A], policy: Policy[S, A], initialStates: 'list[S]', k: float, realQ: 'list[list[float]]') -> 'list[list[float]]' :
    R = MDP.getRewardMatrix(mdp)
    getActionStateValue = lambda s1, a, s2, Q: RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, R, mdp.discount, s1, a, s2, Q)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    maxQ = 1.0#/(1.0-mdp.discount)
    Q = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    visitCount = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState = initialStates[0]
    accReward = 0.0

    plot_diffs = []
    plot_iter = []
    plot_point_count = 100_000

    # The reward is given by the transition to the terminal state not by the terminal state itself
    # And because no movement is done in the terminal states their values are never updated
    # So they should be set to 0
    for s in mdp.states:
        if (not mdp.isTerminal(s)): continue
        for a in mdp.actions:
            Q[mdp.stateIdx(s)][mdp.actionIdx(a)] = 0

    iterations = 2_000_000
    for i in range(iterations):
        if (mdp.isTerminal(currentState)):
            currentState = rnd.choice(initialStates)

        a = policy(Q, currentState)
        nextState = transitionF(currentState, a)
        reward = getActionStateValue(currentState, a, nextState, Q)
        transitionReward = R[mdp.stateIdx(currentState)][mdp.actionIdx(a)]
        visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + 1
        k = math.log2(visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)])
        learningRate: float = k /(k + (visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)]))
        #learningRate: float = k / (k + math.pow(visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)], 0.95))

        accReward = accReward + transitionReward
        Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = (1.0-learningRate) * Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + learningRate * reward
        currentState = nextState

        if ((i < 1_000_000 and i % 10 == 0) or i % 100 == 0):
            diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))
            plot_diffs.append(diff)
            plot_iter.append(i)

    print("Accumulated reward: %.1f" % accReward)
    plt.plot(plot_iter, plot_diffs, 'o', color='black')
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.xticks(rotation = -45)
    plt.ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout(pad=0.2)
    plt.savefig('qlearn.png')
    diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))
    print ("Final Q-diff: " + str(diff))
    return Q
