from training import QLearningCRM, QLearningCPB, ValueIteration
from environments import GridWorld, OfficeWorld
import RLCore
import numpy as np

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

    print("Value iteration")
    Q1 = ValueIteration.valueIteration(gw)
    Q1 = np.array(Q1)
    #Q1[Q1 <= 0.01] = float('nan')
    V1 = RLCore.QtoV(Q1)  # type: ignore
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)  # type: ignore


    iterations = 10_000_000
    print("CRM {} iterations".format(iterations))
    Q = QLearningCRM.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), s1, u1, iterations)
    Q = np.array(Q)
    Q[Q >= 19.9] = float('nan')

    
    OfficeWorld.printVs(of, Q)
    OfficeWorld.printActions(of, Q)

    print("")
    print("CPB")

    Q = QLearningCPB.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), s1, u1, iterations)
    Q = np.array(Q)
    Q[Q >= 19.9] = float('nan')

    OfficeWorld.printVs(of, Q)
    OfficeWorld.printActions(of, Q)
    