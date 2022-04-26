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

    for _ in range(100_000):
        if (mdprm.isTerminal(u)):
            s = s0
            u = u0

        a: A = policy(Q, s, u)
        nextS: S = transitionF(s, a)
        nextU, transitionReward = mdprm.rewardTransition(u, mdprm.labelingFunction(s, a, nextS))

        visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] += 1
        k = 10.0
        learningRate: float = k /(k + (visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)]))
        # print("CRM")
        for u_overline in mdprm.reward_states:
            if (mdprm.isTerminal(u_overline)):
                continue
            nextU_overline, r_overline = mdprm.rewardTransition(u_overline, mdprm.labelingFunction(s, a, nextS))
            if (mdprm.isTerminal(nextU_overline)):
                newValueTerminal = r_overline
                newValueActual = mdprm.discount * r_overline
                # We do this so it looks pretty so that the terminal state will have the value 1 and the one before will have the value
                # 1 * discount, if we do it exactly as it is done in the paper the reward is given at the transition to the terminal state
                # and thus the final result looks a bit odd
                if (s != nextS): # Setting the value for the state that we are in prior to the terminal state
                    Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)] = learningRate * newValueActual + (1.0-learningRate) * Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)]
                for a1 in mdprm.actions: # Setting the value for the actual terminal state
                    Q[mdprm.stateIdx(nextS)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a1)] = learningRate * newValueTerminal + (1.0-learningRate) * Q[mdprm.stateIdx(nextS)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a1)]
            else:
                newValue = r_overline + mdprm.discount * max(Q[mdprm.stateIdx(nextS)][mdprm.rewardStateIdx(nextU_overline)])
                Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)] = learningRate * newValue + (1.0-learningRate) * Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)]
        accReward += transitionReward
        s = nextS
        u = nextU
    
    print("Accumulated reward: %.1f" % accReward)
    return Q
