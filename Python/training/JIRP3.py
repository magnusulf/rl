from copy import deepcopy
from typing import Any, Callable, List, TypeVar
import mdprm
import RLCore
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math

from environments.InferredRM import inferredRM
from environments import GridWorld, OfficeWorld

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



def CRM_episode(mdprm: inferredRM, realRewardTransition, policy: Policy[S, U, A], initialS: 'list[S]', initialRealU, Q, visitCount, iterations: int):
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdprm.baseTransition)
    # maxQ: float = 1.0/(1.0-mdprm.discount)
    # Q = [[[maxQ for _ in mdprm.actions] for _ in mdprm.reward_states] for _ in mdprm.states]
    
    accReward = 0.0
    plot_Qs = []
    plot_iter = []

    s: S = rnd.choice(initialS)
    u = mdprm.dfa._start_state.name
    realU = initialRealU

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
        
        if (mdprm.base_model.isTerminal(realU) or i == 0):
            s = rnd.choice(initialS)
            u = mdprm.dfa._start_state.name
            realU = initialRealU
            # reset trace
            trace = []

        a: A = policy(Q, s, u)
        nextS: S = transitionF(s, a)
        # use actual mdprm for this
        nextU, predictedReward = mdprm.rewardTransition(u, mdprm.labelingFunction(s, a, nextS))
        nextRealU, transitionReward = realRewardTransition(realU, mdprm.labelingFunction(s, a, nextS))

        trace.append(Experience(mdprm.labelingFunction(s, a, nextS), transitionReward))
        if predictedReward != transitionReward:
            example = ("".join([mdprm.lang.fromLabel(e.l) for e in trace]))
            if len(example) > 0 and len(example) <= 8 and True:
                tracecopy = deepcopy(trace)
                X.append(tracecopy)
            
        visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)] += 1

        k = 20.0

        learningRate: float = k /(k + (visitCount[mdprm.stateIdx(s)][mdprm.actionIdx(a)]))
        for u_overline in mdprm.reward_states:
            if (mdprm.isTerminal(u_overline)):
                continue
            nextU_overline, r_overline = mdprm.rewardTransition(u_overline, mdprm.labelingFunction(s, a, nextS))
            newValue = r_overline + mdprm.discount * max(Q[mdprm.stateIdx(nextS)][mdprm.rewardStateIdx(nextU_overline)])
            if (mdprm.isTerminal(nextU_overline)): newValue = r_overline
            Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)] = learningRate * newValue + (1.0-learningRate) * Q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u_overline)][mdprm.actionIdx(a)]
        accReward += transitionReward

        #plot
        if (i % 100 == 0):
            plot_iter.append(i)
            plot_Qs.append(np.array(Q))

        # transition
        s = nextS
        u = nextU
        realU = nextRealU
    return (Q, X, plot_Qs, plot_iter)

def qLearn(base_mdprm: mdprm.mdprm[S, A, any], epsilon: float, initialS: 'list[S]', initialU, num_episodes: int, episode_length, max_trace_len: int, real_dfa_regex):
    #infer initial hypothesis
    X: list[list[Experience]] = []
    hypothesis = inferredRM(base_mdprm)
    hypothesis.inferRewardMachine(X, max_trace_len)
    #initialize Q and visitCount
    maxQ: float = 1.0/(1.0-hypothesis.discount)
    Q = [[[maxQ for _ in hypothesis.actions] for _ in hypothesis.reward_states] for _ in hypothesis.states]
    visitCount = [[0 for _ in hypothesis.actions] for _ in hypothesis.states]
    #policy
    policyEG = policyEpsilonGreedy(hypothesis, epsilon)
    policyR = policyRandom(hypothesis.actions)

    plot_Qs = []
    plot_iter = []

    for episode in range(num_episodes):
        if (hypothesis.dfa.to_regex() == "Ø*"): #no reward machine
            policy = policyR
            #print("using random")
        else:
            policy = policyEG
            #print("using epsilon-greedy with RM: ", hypothesis.dfa.to_regex())

        Q, new_X, inner_plotQs, inner_plot_iter = CRM_episode(hypothesis, base_mdprm.rewardTransition, policy, initialS, initialU, Q, visitCount, episode_length)

        # plot 
        plot_iter = plot_iter + [(episode * episode_length) + i for i in inner_plot_iter]
        if (len(hypothesis.dfa.states) == len(base_mdprm.reward_states)):
            plot_Qs = plot_Qs + inner_plotQs
            
        # infer new hypothesis
        if len(new_X) > 0:
            X = X + new_X
            prevrewardmachine = hypothesis.dfa.to_regex()
            hypothesis.inferRewardMachine(X, max_trace_len)
            # re-init Q and visitCount
            if (prevrewardmachine != hypothesis.dfa.to_regex()):
                Q = [[[maxQ for _ in hypothesis.actions] for _ in hypothesis.reward_states] for _ in hypothesis.states]
                visitCount = [[0 for _ in hypothesis.actions] for _ in hypothesis.states]
    
    if (hypothesis.dfa.to_regex() != real_dfa_regex):
        print("true RM never inferred", hypothesis.dfa.to_regex())
    return (Q, plot_Qs, plot_iter)