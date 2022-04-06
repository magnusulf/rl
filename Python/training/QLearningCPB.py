from typing import Any, Callable, List, TypeVar
import mdprm
import RLCore
import random as rnd
import numpy as np

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')

Policy = Callable[[List[List[List[float]]], S, U], A]

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
        

def qLearn(mdprm: mdprm.mdprm[S, A, U], policy: Policy[S, U, A], s0: S, u0: U) -> 'list[list[list[float]]]' :
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdprm.baseTransition)
    maxQ: float = 1.0/(1.0-mdprm.discount)
    Q = [[[maxQ for _ in mdprm.actions] for _ in mdprm.reward_states] for _ in mdprm.states]
    visitCount = [[0 for _ in mdprm.actions] for _ in mdprm.states]
    accReward = 0.0

    s: S = s0
    u: U = u0

    for i in range(1_000_000):
        if (mdprm.isTerminal(u)):
            s = s0
            u = u0

        a: A = policy(Q, s, u)
        nextS: S = transitionF(s, a)
        nextU: U = mdprm.rewardTransition(u, mdprm.labelingFunction(s, a, nextS))

        visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] = visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] + 1
        k = 10.0
        learningRate: float = k /(k + (visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)]))

        for u_overline in mdprm.reward_states:
            if (mdprm.isTerminal(u_overline)):
                continue
            r_overline: float = mdprm.reward(u_overline, s, a, nextS)
            nextU_overline: U = mdprm.rewardTransition(u_overline, mdprm.labelingFunction(s, a, nextS))
            newValue = 0
            if (mdprm.isTerminal(nextU_overline)):
                newValue = r_overline
            else:
                newValue = r_overline + mdprm.discount * max(Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)])
            Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)] = learningRate * newValue + (1-learningRate) * Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)]
        transitionReward = mdprm.reward(u, s, a, nextS)
        accReward += transitionReward
        s = nextS
        u = nextU
    
    print("Accumulated reward: %.1f" % accReward)
    return Q
