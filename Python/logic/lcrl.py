from typing import Callable, Generic, Tuple, TypeVar
import RLCore
from logic.formula import *
import MDP
from logic.gMonitor import *
import random as rnd
import numpy as np

S = TypeVar('S')
A = TypeVar('A')
Policy = Callable[[List[List[List[float]]], S, ACstate], A]
SQ = Tuple[S, ACstate]

def policyEpsilonGreedy(mdp: MDP.mdp[S, A], asc: AcceptingSubComponent, epsilon: float) -> 'Policy[S, A]' :
    def pol(Q: 'list[list[list[float]]]', s: S, q: ACstate):
        rand = rnd.random()
        if (rand < epsilon): # Random when less than epsilon
            return mdp.actions[rnd.randint(0, len(mdp.actions)-1)]
        else:
            Qvalues = Q[mdp.stateIdx(s)][asc.stateIndex(q)]
            idx = np.argmax(Qvalues)
            return mdp.actions[idx]
    return pol

class CrossProduct(Generic[S, A]):
    def __init__(self, mdp: MDP.mdp[S,A], asc: AcceptingSubComponent):
        self.letters = asc.letters
        self.mdp = mdp
        self.asc = asc
        self.states: List[SQ] = [(s,q) for s in mdp.states for q in asc.states]

    def statecount(self) -> int:
        return len(self.mdp.states) * len(self.asc.states)

    def initialisePsp(self) -> List[List[float]]:
        ret = [[1.0 for _ in self.asc.states] for _ in self.mdp.states]

        for s in self.mdp.states:
            for q in self.asc.states:
                if (not isSink(q)): continue
                ret[self.mdp.stateIdx(s)][self.asc.stateIndex(q)] = 0.0
                

        return ret


def reward(q: ACstate, accSet: Set[ACstate]) -> float:
    if (q in accSet):
        return 1.0
    return 0.0

def lcrlLearn(mdp: MDP.mdp[S, A], asc: AcceptingSubComponent, policy: Policy[S, A], initialStates: 'list[S]', k: float, iterations: int) -> 'list[list[list[float]]]' :
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    cp = CrossProduct(mdp, asc)
    psp = cp.initialisePsp()
    accSet = cp.asc.initialiseA()
    print("accSet")
    print(" : ".join({toStrACstate(x) for x in accSet}))

    visit_phi = [[[0 for _ in mdp.states] for _ in mdp.actions] for _ in mdp.states]
    visit_lr = [[[0 for _ in mdp.actions] for _ in asc.states] for _ in mdp.states]
    visit_psi = [[1 for _ in mdp.actions] for _ in mdp.states]

    def transitionProb(s1: S, q1: ACstate, a: A, s2: S, q2: ACstate) -> float:
        labels = mdp.labelingFunction(s1, a, s2)
        # Check that transition in ASC is sound
        if (asc.transition(q1, labels) != q2):
            return 0

        # Then get probability for transition in the mdp
        mdpProb = visit_phi[mdp.stateIdx(s1)][mdp.actionIdx(a)][mdp.stateIdx(s2)] / visit_psi[mdp.stateIdx(s1)][mdp.actionIdx(a)]
        return mdpProb

    maxQ = 1.0#/(1.0-mdp.discount)
    Q = [[[0.001 for _ in mdp.actions] for _ in asc.states] for _ in mdp.states]
    
    # for q in asc.states:
    #     for s in mdp.states:
    #         for a in mdp.actions:
    #             if (mdp.isTerminal(s) or isSink(q)):
    #                 Q[mdp.stateIdx(s)][asc.stateIndex(q)][mdp.actionIdx(a)] = 0
    #                 # argmax_Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)] = 0
    #             # if (mdprm.isBlocked(s)):
    #             #     visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] = 1

    s: S = rnd.choice(initialStates)
    q: ACstate = asc.q0

    for i in range(iterations):
        if (i % 100_000 == 0):
            print(i)
        if (mdp.isTerminal(s) or isSink(q) or len(accSet) == 0 or i % 10_000 == 0):
            s = rnd.choice(initialStates)
            q = asc.randomQ()
            accSet = cp.asc.initialiseA()
        
        a: A = policy(Q, s, q)
        visit_psi[mdp.stateIdx(s)][mdp.actionIdx(a)] += 1
        nextS = transitionF(s, a)
        labels = mdp.labelingFunction(s, a, nextS)
        nextQ = asc.transition(q, frozenset(labels))
        if (visit_psi[mdp.stateIdx(s)][mdp.actionIdx(a)] == 2):
            visit_phi[mdp.stateIdx(s)][mdp.actionIdx(a)][mdp.stateIdx(nextS)] = 2
        else:
            visit_phi[mdp.stateIdx(s)][mdp.actionIdx(a)][mdp.stateIdx(nextS)] += 1
        r = reward(nextQ, accSet)
        accSet = asc.Acc(nextQ, accSet)
        visit_lr[mdp.stateIdx(s)][asc.stateIndex(q)][mdp.actionIdx(a)] += 1
        learningRate: float = k /(k + (visit_lr[mdp.stateIdx(s)][asc.stateIndex(q)][mdp.actionIdx(a)]))
        newQValue = r + mdp.discount * max(Q[mdp.stateIdx(nextS)][asc.stateIndex(nextQ)])
        Q[mdp.stateIdx(s)][asc.stateIndex(q)][mdp.actionIdx(a)] = learningRate * newQValue + (1.0-learningRate) * Q[mdp.stateIdx(s)][asc.stateIndex(q)][mdp.actionIdx(a)]
        #if (r != 0):
            #print("REWARD")
            #print(" : ".join({toStrACstate(x) for x in accSet}))
        #psp_val = max([sum([psp[mdp.stateIdx(s2)][asc.stateIndex(q2)] * transitionProb(s, q, a, s2, q2) for (s2, q2) in cp.states]) for a in mdp.actions])
        #psp[mdp.stateIdx(s)][asc.stateIndex(q)] = psp_val
        s = nextS
        q = nextQ

    return Q
