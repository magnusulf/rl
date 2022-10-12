from ast import Call, Tuple
from typing import Any, Callable, Generic, List, TypeVar
import MDP
import math
import RLCore
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

S = TypeVar('S')
A = TypeVar('A')

Policy = Callable[[List[List[float]], S], A]
LearningRate = Callable[[int, int], float]
InitialStateSupplier = Callable[[], S]

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

def stateSupplierFixed(states: List[S]) -> InitialStateSupplier[S]:
    return lambda : states[0]

def stateSupplierRandom(states: List[S]) -> InitialStateSupplier[S]:
    return lambda : rnd.choice(states)

def learningRateAvg() -> LearningRate:
    def ret(visitCount: int, t: int) -> float:
        return 1 /(visitCount)
    return ret

def learningRateK(k: int) -> LearningRate:
    def ret(visitCount: int, t: int) -> float:
        return k /(k + visitCount)
    return ret

def learningRateDivideT() -> LearningRate:
    def ret(visitCount: int, t: int) -> float:
        ret = 1 /(t+1)
        return ret
    return ret

def learningRateCta(c: float, a: float) -> LearningRate:
    def ret(visitCount: int, t: int) -> float:
        ret = c /(math.pow(t+1, a))
        return ret
    return ret

def learningRateLog2() -> LearningRate:
    def ret(visitCount: int, t: int) -> float:
        k = math.log2(visitCount)
        return k /(k + visitCount)
    return ret

def learningRateFixed(fixed: float) -> LearningRate:
    def ret(_: int, t: int) -> float:
        return fixed
    return ret

class qLearnSetting(Generic[S, A]):
    def __init__(self, policy: Policy[S, A], lr: LearningRate, stateSupplier: InitialStateSupplier[S], desc: str, color: str) -> None:
        self.policy = policy
        self.learningRate = lr
        self.stateSupplier = stateSupplier
        self.desc = desc
        self.color = color

def qLearn(mdp: MDP.mdp[S, A], qls: qLearnSetting[S,A], iterations: int, realQ: 'list[list[float]]') -> 'tuple[list[list[float]], list[float], list[float], list[int], int]' :
    R = MDP.getRewardMatrix(mdp)
    getActionStateValue = lambda s1, a, s2, Q: RLCore.getActionToStateValue(mdp.stateIdx, mdp.actionIdx, R, mdp.discount, s1, a, s2, Q)
    transitionF = RLCore.baseTransitionFunctionToStochasticTransitionFunction(mdp.baseTransition)
    maxQ = 1.0/(1.0-mdp.discount)
    Q = [[maxQ for _ in mdp.actions] for _ in mdp.states]
    visitCount = [[0 for _ in mdp.actions] for _ in mdp.states]
    currentState: S = qls.stateSupplier()
    accReward = 0.0

    plot_Qdiffs: List[float] = []
    plot_polDiffs: List[float] = []
    plot_iter: List[int] = []
    cover_time = -1
    plot_point_count = math.floor(iterations / 1000)

    # The reward is given by the transition to the terminal state not by the terminal state itself
    # And because no movement is done in the terminal states their values are never updated
    # So they should be set to 0
    for s in mdp.states:
        if (not mdp.isTerminal(s)): continue
        for a in mdp.actions:
            Q[mdp.stateIdx(s)][mdp.actionIdx(a)] = 0
            visitCount[mdp.stateIdx(s)][mdp.actionIdx(a)] = 1

    for i in range(iterations):
        if (mdp.isTerminal(currentState)):
            currentState = qls.stateSupplier()

        a = qls.policy(Q, currentState)
        nextState = transitionF(currentState, a)
        reward = getActionStateValue(currentState, a, nextState, Q)
        transitionReward = R[mdp.stateIdx(currentState)][mdp.actionIdx(a)]
        visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + 1
        learningRate: float = qls.learningRate(visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)], i)

        if (cover_time == -1 and visitCount[mdp.stateIdx(currentState)][mdp.actionIdx(a)] == 1 and np.min(visitCount) > 0):
            cover_time = i

        accReward = accReward + transitionReward
        Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] = (1.0-learningRate) * Q[mdp.stateIdx(currentState)][mdp.actionIdx(a)] + learningRate * reward
        currentState = nextState

        if (i % plot_point_count == 0):
            Qdiff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))
            plot_Qdiffs.append(Qdiff)

            polDiff = 0
            for s in mdp.states:
                if (mdp.isTerminal(s)):
                    continue
                policy_action = np.argmax(Q[mdp.stateIdx(s)])
                policy_val = realQ[mdp.stateIdx(s)][policy_action]
                optimal_val = np.max(realQ[mdp.stateIdx(s)])
                if (abs(policy_val - optimal_val) > 0.01):
                    polDiff += 1
            plot_polDiffs.append(polDiff)

            plot_iter.append(i)
        


    # print("Accumulated reward: %.1f" % accReward)
    # plt.plot(plot_iter, plot_diffs, 'o', color='black')
    # plt.xlabel('Iterations')
    # plt.ylabel('Q diff')
    # plt.xticks(rotation = -45)
    # plt.ticklabel_format(useOffset=False, style='plain')
    # plt.tight_layout(pad=0.2)
    # plt.savefig('qlearn.png')
    # diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))
    # print ("Final Q-diff: " + str(diff))
    return (Q, plot_Qdiffs, plot_polDiffs, plot_iter, cover_time)


def qLearnMany(mdp: MDP.mdp[S, A], settings: List[qLearnSetting[S,A]], iterations, realQ, iter2):
    plot_iters = []
    mainplot_polMeanDiff = []
    mainplot_polErrors = []
    mainplot_QmeanDiff = []
    mainplot_Qerrors = []
    
    for qls in settings:
        Qs = [] 
        plot_Qdiffss = []
        plot_polDiffss = []
        cover_times = []
        diffs = []
        for j in range(iter2):
            (Q, plot_Qdiffs, plot_polDiffs, plot_iter, cover_time) = qLearn(mdp, qls, iterations, realQ)
            diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

            Qs.append(Q)
            plot_Qdiffss.append(plot_Qdiffs)
            plot_polDiffss.append(plot_polDiffs)
            plot_iters = plot_iter
            cover_times.append(cover_time)
            diffs.append(diff)
            print(qls.desc + " ("+ str(j+1) + ")")

        plot_polMeanDiff = np.mean(np.array(plot_polDiffss), axis=0)
        plot_polErrors = np.std(np.array(plot_polDiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

        plot_QmeanDiff = np.mean(np.array(plot_Qdiffss), axis=0)
        plot_Qerrors = np.std(np.array(plot_Qdiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

        mainplot_polMeanDiff.append(plot_polMeanDiff)
        mainplot_polErrors.append(plot_polErrors)
        mainplot_QmeanDiff.append(plot_QmeanDiff)
        mainplot_Qerrors.append(plot_Qerrors)

        print ("Final Q-diff mean (" + qls.desc + "): " + str(np.mean(diffs)))
        print ("Cover time median (" + qls.desc + "): " + str(np.median(cover_times)))

        # Singleplot
        plt.clf()   
        plt.errorbar(plot_iters, plot_polMeanDiff, yerr=plot_polErrors, marker='.', color=qls.color, ecolor='grey', markersize=1, label=qls.desc)
        plt.title("Q learning average " + mdp.desc + " (n= " + str(iter2) + ")")
        plt.xlabel('Iterations')
        plt.ylabel('Policy diff #')
        #plt.yscale('log')
        plt.ylim([0.0, len([x for x in mdp.states if not mdp.isTerminal(x)])])
        plt.xticks(rotation = -45)
        plt.legend()
        plt.tight_layout(pad=0.2)
        plt.savefig('Policy diff ' + mdp.desc + " " +  qls.desc + ' (n= ' + str(iter2) + ').png')

        plt.clf()   
        plt.errorbar(plot_iters, plot_QmeanDiff, yerr=plot_Qerrors, marker='.', color=qls.color, ecolor='grey', markersize=1, label=qls.desc)
        plt.title("Q learning average " + mdp.desc + " (n= " + str(iter2) + ")")
        plt.xlabel('Iterations')
        plt.ylabel('Q diff')
        plt.yscale('log')
        plt.ylim([0.01, 1/(1-mdp.discount)])
        plt.xticks(rotation = -45)
        plt.legend()
        plt.tight_layout(pad=0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('Q-diff ' + mdp.desc + " " + qls.desc + ' (n= ' + str(iter2) + ').png')

    
    # main plot
    plt.clf()
    for i in range(len(settings)):
        qls = settings[i]
        plt.errorbar(plot_iters, mainplot_polMeanDiff[i], yerr=mainplot_polErrors[i], marker='.', color=qls.color, ecolor='grey', markersize=1, label=qls.desc)
    plt.title('Q learning average ' + mdp.desc + ' (n= ' + str(iter2) + ')')
    plt.xlabel('Iterations')
    plt.ylabel('Policy diff #')
    #plt.yscale('log')
    plt.ylim([0.0, len([x for x in mdp.states if not mdp.isTerminal(x)])])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('Policy diff ALL ' + mdp.desc + ' (n= ' + str(iter2) + ').png')

    plt.clf()   
    for i in range(len(settings)):
        qls = settings[i]
        plt.errorbar(plot_iters, mainplot_QmeanDiff[i], yerr=mainplot_Qerrors[i], marker='.', color=qls.color, ecolor='grey', markersize=1, label=qls.desc)
    plt.title('Q learning average ' + mdp.desc + ' (n= ' + str(iter2) + ')')
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdp.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('Q-diff ALL ' + mdp.desc + ' (n= ' + str(iter2) + ').png')

        
