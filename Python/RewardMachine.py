from training import QLearningCRM, QLearningCPB, ValueIteration, ValueIterationRM
from environments import GridWorld, OfficeWorld
import RLCore
import numpy as np
import mdprm
import matplotlib.pyplot as plt
import math



def qLearnMany(mdprm, policy, initialS, initialU, iterations, realQ, iter2):
    crm_Qs = [] 
    crm_plot_diffss = []
    plot_iters = []
    crm_cover_times = []
    crm_diffs = []

    for j in range(iter2):
        (Q, plot_diffs, plot_iter, cover_time) = QLearningCRM.qLearn(mdprm, policy, initialS, initialU, iterations, realQ, j)
        diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

        crm_Qs.append(Q)
        crm_plot_diffss.append(plot_diffs)
        plot_iters = plot_iter
        crm_cover_times.append(cover_time)
        crm_diffs.append(diff)

    crm_plot_diffs = np.mean(np.array(crm_plot_diffss), axis=0)
    crm_plot_errors = np.std(np.array(crm_plot_diffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

    cpb_Qs = [] 
    cpb_plot_diffss = []
    plot_iters = []
    cpb_cover_times = []
    cpb_diffs = []
    for j in range(iter2):
        (Q, plot_diffs, plot_iter, cover_time) = QLearningCPB.qLearn(mdprm, policy, initialS, initialU, iterations, realQ, j)
        diff = np.max(np.abs(np.subtract(np.array(Q), np.array(realQ))))

        cpb_Qs.append(Q)
        cpb_plot_diffss.append(plot_diffs)
        plot_iters = plot_iter
        cpb_cover_times.append(cover_time)
        cpb_diffs.append(diff)

    cpb_plot_diffs = np.mean(np.array(cpb_plot_diffss), axis=0)
    cpb_plot_errors = np.std(np.array(cpb_plot_diffss), axis=0, ddof=1) * 1.96 / math.sqrt(iter2)

    # Both
    plt.clf()   
    plt.errorbar(plot_iters, crm_plot_diffs, yerr=crm_plot_errors, marker='.', color='red', ecolor='grey', markersize=1, label='CRM')
    plt.errorbar(plot_iters, cpb_plot_diffs, yerr=cpb_plot_errors, marker='.', color='blue', ecolor='grey', markersize=1, label='CPB')
    plt.title("Q learning average (n= " + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdprm.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('qlearn avg (n= ' + str(iter2) + ').png')

    # CRM
    plt.clf()   
    plt.errorbar(plot_iters, crm_plot_diffs, yerr=crm_plot_errors, marker='.', color='red', ecolor='grey', markersize=1, label='CRM')
    plt.title("Q learning average (n= " + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdprm.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('qlearn CRM avg (n= ' + str(iter2) + ').png')

    # CPB
    plt.clf()   
    plt.errorbar(plot_iters, cpb_plot_diffs, yerr=cpb_plot_errors, marker='.', color='blue', ecolor='grey', markersize=1, label='CPB')
    plt.title("Q learning average (n= " + str(iter2) + ")")
    plt.xlabel('Iterations')
    plt.ylabel('Q diff')
    plt.yscale('log')
    plt.ylim([0.01, 1/(1-mdprm.discount)])
    plt.xticks(rotation = -45)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig('qlearn CPB avg (n= ' + str(iter2) + ').png')

    print ("Final CRM  Q-diff mean: " + str(np.mean(crm_diffs)))
    print ("cover time CRM median: " + str(np.median(crm_cover_times)))

    print ("Final CPB  Q-diff mean: " + str(np.mean(cpb_diffs)))
    print ("cover time CPB median: " + str(np.median(cpb_cover_times)))

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
    gw = GridWorld.gridworld(of.max_x, of.max_y, of.blocked_states, 
    {(5, 5): 1, (1, 5) : 0, (1, 9) : 0, (5,1) : 0, (5, 9) : 0, (9, 1) : 0, (9, 5) : 0, (9,9) : 0}, s1, of.turnProb, of.discount)

    print("Value iteration RM")
    Qvi = ValueIterationRM.valueIteration(of)
    Qvi = np.array(Qvi)

    OfficeWorld.printVs(of, Qvi)
    OfficeWorld.printActions(of, Qvi)

    print("")
    iterations = 1_500_000
    print("CRM {} iterations".format(iterations))
    qLearnMany(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), [s1, (9, 0), (9, 10)], [u1, 'coffee'], iterations, Qvi, 201)
    #Q = np.array(Q)
    #OfficeWorld.printVs(of, Q)
    #OfficeWorld.printActions(of, Q)

    print("")
    # print("CPB")

    # Q = QLearningCPB.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), s1, u1, iterations, Qvi)
    # Q = np.array(Q)

    # OfficeWorld.printVs(of, Q)
    # OfficeWorld.printActions(of, Q)
