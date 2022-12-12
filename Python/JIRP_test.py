
from typing import TypeVar
from training import QLearningCRM, QLearningCPB, ValueIteration, ValueIterationRM, JIRP3
from environments import GridWorld, OfficeWorld, RiverSwimRM
import RLCore
import numpy as np
import mdprm
import matplotlib.pyplot as plt
import math
from datetime import datetime

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')


def qLearnMany(mdprm: mdprm.mdprm[S,A,U], policy, initialS: 'list[S]', initialU: 'list[U]', iterations: int, realQ, real_dfa_regex: str, iter2: int):
    # crm_Qs = [] 
    # crm_plot_polDiffss = []
    # crm_plot_Qdiffss = []
    # plot_iters = []
    # crm_cover_times = []
    # crm_diffs = []

    # for j in range(iter2):
    #     (Q, plot_Qdiffs, plot_polDiffs, plot_iter, cover_time) = QLearningCRM.qLearn(mdprm, policy, initialS, initialU, iterations, realQ, j)
    #     diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

    #     crm_Qs.append(Q)
    #     crm_plot_polDiffss.append(plot_polDiffs)
    #     crm_plot_Qdiffss.append(plot_Qdiffs)
    #     plot_iters = plot_iter
    #     crm_cover_times.append(cover_time)
    #     crm_diffs.append(diff)
    #     print('CRM ' + mdprm.desc + " " + str(j))

    # crm_plot_Qdiffs = np.mean(np.array(crm_plot_Qdiffss), axis=0)
    # crm_plot_Qerrors = np.std(np.array(crm_plot_Qdiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)
    # crm_plot_polDiffs = np.mean(np.array(crm_plot_polDiffss), axis=0)
    # crm_plot_polErrors = np.std(np.array(crm_plot_polDiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

    cpb_Qs = [] 
    cpb_plot_polDiffss = []
    cpb_plot_Qdiffss = []
    cpb_plot_iters = []
    cpb_cover_times = []
    cpb_diffs = []
    for j in range(iter2):
        (Q, plot_Qdiffs, plot_polDiffs, plot_iter, cover_time) = QLearningCPB.qLearn(mdprm, policy, initialS, initialU, iterations, realQ, j)
        diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

        cpb_Qs.append(Q)
        cpb_plot_polDiffss.append(plot_polDiffs)
        cpb_plot_Qdiffss.append(plot_Qdiffs)
        cpb_plot_iters = plot_iter
        cpb_cover_times.append(cover_time)
        cpb_diffs.append(diff)
        print('CPB ' + mdprm.desc + " "+ str(j)  + " time: " + datetime.now().strftime("%H:%M:%S"))

    cpb_plot_Qdiffs = np.mean(np.array(cpb_plot_Qdiffss), axis=0)
    cpb_plot_Qerrors = np.std(np.array(cpb_plot_Qdiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)
    cpb_plot_polDiffs = np.mean(np.array(cpb_plot_polDiffss), axis=0)
    cpb_plot_polErrors = np.std(np.array(cpb_plot_polDiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

    jirp_Qs = [] 
    jirp_plot_polDiffss = []
    jirp_plot_Qdiffss = []
    jirp_plot_iters = []
    #jirp_cover_times = []
    #jirp_diffs = []


    for j in range(iter2):
        episode_length = 1000
        num_episodes = math.ceil(iterations / episode_length)
        max_trace_len = 6
        real_dfa_regex
        (Q, plot_Qdiffs, plot_polDiffs, plot_iter) = JIRP3.qLearn(mdprm, 0.2, initialS, initialU[0], num_episodes, episode_length, max_trace_len, real_dfa_regex, realQ)
        #diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

        jirp_Qs.append(Q)
        jirp_plot_polDiffss.append(plot_polDiffs)
        jirp_plot_Qdiffss.append(plot_Qdiffs)
        jirp_plot_iters = plot_iter
        #jirp_cover_times.append(cover_time)
        #jirp_diffs.append(diff)
        print('JIRP ' + mdprm.desc + " " + str(j)  + " time: " +  datetime.now().strftime("%H:%M:%S"))

    jirp_plot_Qdiffs = np.mean(np.array(jirp_plot_Qdiffss), axis=0)
    jirp_plot_Qerrors = np.std(np.array(jirp_plot_Qdiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)
    jirp_plot_polDiffs = np.mean(np.array(jirp_plot_polDiffss), axis=0)
    jirp_plot_polErrors = np.std(np.array(jirp_plot_polDiffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

    plt.clf()
    plt.errorbar(cpb_plot_iters, cpb_plot_Qdiffs, yerr=cpb_plot_Qerrors, marker='o', color='blue', ecolor='grey', markersize=1, label='CPB')
    plt.errorbar(jirp_plot_iters, jirp_plot_Qdiffs, yerr=jirp_plot_Qerrors, marker='o', color='red', ecolor='grey', markersize=1, label='JIRP')
    plt.title('Q learning average ' + mdprm.desc + ' (n=' + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdprm.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    #plt.tight_layout(pad=0.2)
    plt.savefig('qlearn Q-diff  ' + mdprm.desc + ' (n=' + str(iter2) + ').png', bbox_inches='tight')

    # Both
    plt.clf()   
    plt.errorbar(cpb_plot_iters, cpb_plot_polDiffs, yerr=cpb_plot_polErrors, marker='o', color='blue', ecolor='grey', markersize=1, label='CPB')
    plt.errorbar(jirp_plot_iters, jirp_plot_polDiffs, yerr=jirp_plot_polErrors, marker='o', color='red', ecolor='grey', markersize=1, label='JIRP')
    plt.title('Q learning average  ' + mdprm.desc + ' (n=' + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Policy diff #')
    plt.ylim([0.0, len(mdprm.states) * len([x for x in mdprm.reward_states if not mdprm.isTerminal(x)])])
    plt.xticks(rotation = -45)
    plt.legend()
    #plt.tight_layout(pad=0.3)
    plt.savefig('qlearn policy # ' + mdprm.desc + ' (n=' + str(iter2) + ').png', bbox_inches='tight')

    # # CRM
    # plt.clf()   
    # plt.errorbar(plot_iters, crm_plot_diffs, yerr=crm_plot_errors, marker='.', color='red', ecolor='grey', markersize=1, label='CRM')
    # plt.title("Q learning average (n= " + str(iter2) + ")")
    # plt.xlabel('Iterations')
    # plt.ylabel('Policy diff #')
    # #plt.yscale('log')
    # plt.ylim([0.0, len(mdprm.states) * len(mdprm.reward_states)])
    # plt.xticks(rotation = -45)
    # plt.legend()
    # plt.tight_layout(pad=0.2)
    # plt.savefig('qlearn CRM policy # (n= ' + str(iter2) + ').png')

    # # CPB
    # plt.clf()   
    # plt.errorbar(plot_iters, cpb_plot_diffs, yerr=cpb_plot_errors, marker='.', color='blue', ecolor='grey', markersize=1, label='CPB')
    # plt.title("Q learning average (n= " + str(iter2) + ")")
    # plt.xlabel('Iterations')
    # plt.ylabel('Policy diff #')
    # #plt.yscale('log')
    # plt.ylim([0.0, len(mdprm.states) * len(mdprm.reward_states)])
    # plt.xticks(rotation = -45)
    # plt.legend()
    # plt.tight_layout(pad=0.2)
    # plt.savefig('qlearn CPB policy # (n= ' + str(iter2) + ').png')

    
    #print ("cover time CRM median: " + str(np.median(crm_cover_times)))

    print ("Final CPB  Q-diff mean: " + str(np.mean(cpb_diffs))  + datetime.now().strftime("%H:%M:%S"))
    #print ("Final CRM  Q-diff mean: " + str(np.mean(crm_diffs)))
    #print ("cover time CPB median: " + str(np.median(cpb_cover_times)))

if __name__ == '__main__':

    of = OfficeWorld.officeworld(11, 11,
        [(3,0), (3,2), (7,0), (7,2), (0,3), (2,3), (2,7), (0,7), (8, 3), (8, 7), (10, 3), (10, 7), (3, 8), (3, 10), (7, 8), (7, 10),
        (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (3, 4), (3, 5), (3, 6), (3, 7), (7, 4), (7, 5), (7, 6), (7, 7), (4, 7), (6, 7)],
        [(5,5)], # Office
        [(4,8), (4, 2)], # Coffes
        [(1, 5), (1, 9), (5,1), (5, 9), (9, 1), (9, 5), (9,9)], # Decorations
        0.1, # turn probability
        0.95) # discount
    s1 = (1, 2)
    u1 = 'start'

    smallof = OfficeWorld.officeworld(3, 3,
    [],
    [(1,1)], # Office
    [(1,0), (0, 2)], # Coffes
    [], # Decorations
    0.1, # turn probability
    0.95) # discount
    s1 = (0, 0)
    u1 = 'start'

    gw = GridWorld.gridworld(of.max_x, of.max_y, of.blocked_states, 
    {(5, 5): 1, (1, 5) : 0, (1, 9) : 0, (5,1) : 0, (5, 9) : 0, (9, 1) : 0, (9, 5) : 0, (9,9) : 0}, s1, of.turnProb, of.discount)

    rs = RiverSwimRM.riverswimRM(8, 0.55, 0.05, 0.1, 1.0, 0.95)

    env = smallof

    print("Value iteration RM: " + datetime.now().strftime("%H:%M:%S"))
    Qvi = ValueIterationRM.valueIteration(env)
    Qvi = np.array(Qvi)
    print("Value iteration finished: " + datetime.now().strftime("%H:%M:%S"))

    print("")
    if (env == of):
        iterations = 2_000_000
        iter2 = 10
        initialStates = [s1, (10,0), (10, 10), (0, 10)]
        initialRewardStates = [u1]

        real_regex = '(cc*(o(o|c))|o)*cc*o'

        OfficeWorld.printVs(of, Qvi)
        OfficeWorld.printActions(of, Qvi)
    elif (env == smallof):
        iterations = 1000_000
        iter2 = 3
        initialStates = [s1]
        initialRewardStates = [u1]

        real_regex = '(cc*(o(o|c))|o)*cc*o'

        OfficeWorld.printVs(smallof, Qvi)
        OfficeWorld.printActions(smallof, Qvi)
    else:
        iterations = 100_000
        iter2 = 1000
        initialStates = [1]
        initialRewardStates = ['leftgoal', 'rightgoal']

        real_regex = ''

        RiverSwimRM.printVs(rs, Qvi)
        RiverSwimRM.printActions(rs, Qvi)
    
    print("CPB {} iterations".format(iterations))

    
    qLearnMany(env, QLearningCRM.policyEpsilonGreedy(env, 0.2), initialStates, initialRewardStates, iterations, Qvi, real_regex, iter2)

    #Q = np.array(Q)
    #OfficeWorld.printVs(of, Q)
    #OfficeWorld.printActions(of, Q)

    print("")
    # print("CPB")

    # Q = QLearningCPB.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), s1, u1, iterations, Qvi)
    # Q = np.array(Q)

    # OfficeWorld.printVs(of, Q)
    # OfficeWorld.printActions(of, Q)
