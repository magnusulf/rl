
from typing import TypeVar
from training import QLearningCRM, QLearningCPB, ValueIteration, ValueIterationRM, JIRP3
from environments import GridWorld, OfficeWorld, RiverSwimRM
import RLCore
import numpy as np
import mdprm
import matplotlib.pyplot as plt
import math
from datetime import datetime
from itertools import permutations

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')


def qLearnMany(mdprm: mdprm.mdprm[S,A,U], policy, initialS: 'list[S]', initialU: 'list[U]', iterations: int, realQ, real_dfa_regex: str, iter2: int, episode_lengths: 'list[int]', colors: 'list[str]'):

    for i, episode_length in enumerate(episode_lengths):

        print("Testing episode length:", episode_length)

        jirp_Qs = [] 
        jirp_plot_polDiffss = []
        jirp_plot_Qdiffss = []
        jirp_plot_iters = []
        jirp_diffs = []

        for j in range(iter2):
            num_episodes = math.ceil(iterations / episode_length)
            max_trace_len = 6
            real_dfa_regex
            (Q, plot_Qs, plot_iter) = JIRP3.qLearn(mdprm, 0.2, initialS, initialU[0], num_episodes, episode_length, max_trace_len, real_dfa_regex)

            maxQ: float = 1.0/(1.0-mdprm.discount)
            default_Q = [[[maxQ for _ in mdprm.actions] for _ in mdprm.reward_states] for _ in mdprm.states]
            plot_Qs = ([default_Q] * (len(plot_iter) - len(plot_Qs))) + plot_Qs
            last_Q = plot_Qs[-1]
            perm_Qs = [np.max(np.abs(np.subtract(np.array(np.array(last_Q)[:,perm,:]), np.array(realQ)))) for perm in permutations(range(len(mdprm.reward_states)))]
            best_perm = list(permutations(range(len(mdprm.reward_states))))[np.argmin(perm_Qs)]
            Q = np.array(last_Q)[:,best_perm,:]
            diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))
            jirp_diffs.append(diff)
            plot_Qs = [np.array(q)[:,best_perm,:] for q in plot_Qs]
            plot_Qdiffs = []
            plot_polDiffs = []
            for q in plot_Qs:
                Qdiff = np.max(np.abs(np.subtract(np.array(q), np.array(realQ))))
                plot_Qdiffs.append(Qdiff)

                diff = 0
                for u in mdprm.reward_states:
                    for s in mdprm.states:
                        if (mdprm.isTerminal(u) or mdprm.isBlocked(s)):
                            continue
                        policy_action = np.argmax(q[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)])
                        policy_val = realQ[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)][policy_action]
                        optimal_val = np.max(realQ[mdprm.stateIdx(s)][mdprm.rewardStateIdx(u)])
                        if (abs(policy_val - optimal_val) > 0.01):
                            diff += 1
                plot_polDiffs.append(diff)

            jirp_Qs.append(Q)
            jirp_plot_polDiffss.append(plot_polDiffs)
            jirp_plot_Qdiffss.append(plot_Qdiffs)
            jirp_plot_iters = plot_iter
            print('JIRP ' + mdprm.desc + " " + str(j)  + " time: " +  datetime.now().strftime("%H:%M:%S"))

        jirp_plot_Qdiffs = np.mean(np.array(jirp_plot_Qdiffss), axis=0)
        jirp_plot_Qerrors = np.std(np.array(jirp_plot_Qdiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)
        jirp_plot_polDiffs = np.mean(np.array(jirp_plot_polDiffss), axis=0)
        jirp_plot_polErrors = np.std(np.array(jirp_plot_polDiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

        plt.figure('q')
        plt.errorbar(jirp_plot_iters, jirp_plot_Qdiffs, yerr=jirp_plot_Qerrors, marker='o', color=colors[i], ecolor='grey', markersize=1, label='episode length: '+str(episode_length))

        plt.figure('p')
        plt.errorbar(jirp_plot_iters, jirp_plot_polDiffs, yerr=jirp_plot_polErrors, marker='o', color=colors[i], ecolor='grey', markersize=1, label='episode length: '+str(episode_length))

        print ("Final JIRP Q-diff mean: " + str(np.mean(jirp_diffs))  + datetime.now().strftime(" %H:%M:%S"))

    plt.figure('q')
    plt.title('Q learning average JIRP in ' + mdprm.desc + ' (n=' + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdprm.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.savefig('qlearn Q-diff  ' + mdprm.desc + ' (n=' + str(iter2) + ').png', bbox_inches='tight')

    plt.figure('p') 
    plt.title('Q learning average JIRP in' + mdprm.desc + ' (n=' + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Policy diff #')
    plt.ylim([0.0, len(mdprm.states) * len([x for x in mdprm.reward_states if not mdprm.isTerminal(x)])])
    plt.xticks(rotation = -45)
    plt.legend()
    #plt.tight_layout(pad=0.3)
    plt.savefig('qlearn policy # ' + mdprm.desc + ' (n=' + str(iter2) + ').png', bbox_inches='tight')

    

if __name__ == '__main__':

    of = OfficeWorld.officeworld(11, 11,
        [(3,0), (3,2), (7,0), (7,2), (0,3), (2,3), (2,7), (0,7), (8, 3), (8, 7), (10, 3), (10, 7), (3, 8), (3, 10), (7, 8), (7, 10),
        (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (3, 4), (3, 5), (3, 6), (3, 7), (7, 4), (7, 5), (7, 6), (7, 7), (4, 7), (6, 7)],
        [(5,5)], # Office
        [(4,8), (4, 2)], # Coffes
        [], # Decorations
        0.1, # turn probability
        0.95) # discount
    s1 = (1, 2)
    u1 = 'start'

    medof = OfficeWorld.officeworld(7, 7,
        [(3,0), (3,2), (0,3), (2,3),
        (3, 3), (4, 3), (5, 3), (6, 3), (3, 4), (3, 6)],
        [(5,5)], # Office
        [(5, 1)], # Coffes
        [], # Decorations
        0.1, # turn probability
        0.95) # discount
    s1 = (0, 0)
    u1 = 'start'

    smallof = OfficeWorld.officeworld(3, 3,
        [],
        [(0,1)], # Office
        [(1,0), (0, 2)], # Coffes
        [], # Decorations
        0.1, # turn probability
        0.95) # discount
    s1 = (0, 0)
    u1 = 'start'

    gw = GridWorld.gridworld(of.max_x, of.max_y, of.blocked_states, 
    {(5, 5): 1, (1, 5) : 0, (1, 9) : 0, (5,1) : 0, (5, 9) : 0, (9, 1) : 0, (9, 5) : 0, (9,9) : 0}, s1, of.turnProb, of.discount)

    rs = RiverSwimRM.riverswimRM(8, 0.55, 0.05, 0.1, 1.0, 0.95)

    env = of
    colors = ['red', 'green', 'blue', 'purple']

    print("Value iteration RM: " + datetime.now().strftime("%H:%M:%S"))
    Qvi = ValueIterationRM.valueIteration(env)
    Qvi = np.array(Qvi)
    print("Value iteration finished: " + datetime.now().strftime("%H:%M:%S"))

    print("")
    if (env == of):
        iterations = 1_000_000
        episode_lengths = [1000, 10_000, 100_000]
        iter2 = 500
        #initialStates = [s1, (10,0), (10, 10), (0, 10)]
        initialStates = [s1]
        initialRewardStates = [u1]

        real_regex = '(cc*(o(o|c))|o)*cc*o'

        OfficeWorld.printVs(of, Qvi)
        OfficeWorld.printActions(of, Qvi)
    elif (env == medof):
        iterations = 1_000_000
        episode_lengths = [1000, 10_000, 100_000]
        iter2 = 10
        initialStates = [s1]
        initialRewardStates = [u1]

        real_regex = '(cc*(o(o|c))|o)*cc*o'

        OfficeWorld.printVs(medof, Qvi)
        OfficeWorld.printActions(medof, Qvi)
    elif (env == smallof):
        iterations = 100_000
        episode_lengths = [1000, 10_000, 50_000]
        iter2 = 3
        initialStates = [s1]
        initialRewardStates = [u1]

        real_regex = '(cc*(o(o|c))|o)*cc*o'

        OfficeWorld.printVs(smallof, Qvi)
        OfficeWorld.printActions(smallof, Qvi)
    else:
        iterations = 100_000
        episode_length = 1000
        iter2 = 1000
        initialStates = [1]
        initialRewardStates = ['leftgoal', 'rightgoal']

        real_regex = ''

        RiverSwimRM.printVs(rs, Qvi)
        RiverSwimRM.printActions(rs, Qvi)
    
    print("Test {} iterations".format(iterations))
    qLearnMany(env, QLearningCRM.policyEpsilonGreedy(env, 0.2), initialStates, initialRewardStates, iterations, Qvi, real_regex, iter2, episode_lengths, colors)
    print("")
