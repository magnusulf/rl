from typing import Any, Callable, List, TypeVar
import mdprm
import RLCore
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')

Policy = Callable[[List[List[List[float]]], S, U], A]

class Experience():
    def __init__(self, labels: List[str], reward: float):
        self.l = labels
        self.r = reward

    def __repr__(self):
        return "({l}, {r})".format(l=self.l, r=self.r)

def policyRandom(actions: 'list[A]') -> 'Callable[[list[list[list[float]]], S, U], A]' :
    def pol(Q: 'list[list[list[float]]]', s: S, u: U) -> A:
        return actions[rnd.randint(0, len(actions)-1)]
    return pol

def policyEpsilonGreedy(mdp: mdprm.mdprm[S, A, U], epsilon: float) -> 'Callable[[list[list[list[float]]], S, U], A]' :
    def pol(Q: 'list[list[list[float]]]', s: S, u: U) -> A:
        rand = rnd.random()
        if (rand < epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            Qvalues = Q[mdp.stateIdx(s)][mdp.rewardStateIdx(u)]
            idx = np.argmax(Qvalues)
            return mdp.actions[idx]
    return pol



def qLearn(mdprm: mdprm.mdprm[S, A, U], policy: Policy[S, U, A], initialS: 'list[S]', initialU: 'list[U]', iterations: int):
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdprm.baseTransition)
    maxQ: float = 1.0/(1.0-mdprm.discount)
    Q = [[[maxQ for _ in mdprm.actions] for _ in mdprm.reward_states] for _ in mdprm.states]
    visitCount = [[0 for _ in mdprm.actions] for _ in mdprm.states]
    accReward = 0.0

    s: S = rnd.choice(initialS)
    u: U = rnd.choice(initialU)

    #argmax_Q = np.argmax(np.array(realQ), axis=2)

    for u in mdprm.reward_states:
        for s in mdprm.states:
            for a in mdprm.actions:
                if (mdprm.isTerminal(u) or mdprm.isBlocked(s)):
                    Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)][mdprm.actionIdx(a)] = 0
                    #argmax_Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)] = 0
                if (mdprm.isBlocked(s)):
                    visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] = 1

    X: list[list[Experience]] = []
    trace: list[Experience] = []

    for i in range(iterations):
        if (mdprm.isTerminal(u)):
            s = rnd.choice(initialS)
            u = rnd.choice(initialU)
            # add trace to X and reset
            X.append(trace)
            trace = []

        a: A = policy(Q, s, u)
        nextS: S = transitionF(s, a)
        nextU, transitionReward = mdprm.rewardTransition(u, mdprm.labelingFunction(s, a, nextS))

        trace.append(Experience(mdprm.labelingFunction(s, a, nextS), transitionReward))

        visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] += 1

        k = 20.0

        learningRate: float = k /(k + (visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)]))
        # print("CRM")
        for u_overline in mdprm.reward_states:
            if (mdprm.isTerminal(u_overline)):
                continue
            nextU_overline, r_overline = mdprm.rewardTransition(u_overline, mdprm.labelingFunction(s, a, nextS))
            newValue = r_overline + mdprm.discount * max(Q[mdprm.stateIdx(nextS)][mdprm.rewardStateIdx(nextU_overline)])
            if (mdprm.isTerminal(nextU_overline)): newValue = r_overline
            Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)] = learningRate * newValue + (1.0-learningRate) * Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)]
        accReward += transitionReward
        s = nextS
        u = nextU

    return (Q, X)

