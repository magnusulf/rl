from environments import GridWorld,  RiverSwim, OfficeWorld
from training import ValueIteration, policyIteration, QLearning, UCBLearning
import RLCore

if __name__ == '__main__':
    #let turnProbability = 0.1
    #let stayProbability = 0.1
    #let livingReward = 0.0

    # This is the office world we later use with reward machines.
    # For the sake of similarity we use a gridworld that is idential
    # Except the only goal is to reach the office
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
    
    #gw = GridWorld.gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.1, 0.50)
    rs = RiverSwim.riverswim(4, 0.55, 0.05, 0.1, 1.0, 0.8)
    
    print("Value iteration")
    #Q = ValueIteration.valueIteration(gw)
    Q1 = ValueIteration.valueIteration(gw)
    V1 = RLCore.QtoV(Q1)
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)
    #RiverSwim.printV(V1)
    #RiverSwim.printActionsFromQ(Q1)

    print("Q learning (e-greedy) (k-plus learning rate)")
    policy = QLearning.policyEpsilonGreedy(gw, 0.2)
    #Q2 = QLearning.qLearn(gw, policy, gw.starting_state) , (5, 2), (9,2), (1, 10), (9, 10)
    Q2 = QLearning.qLearn(gw, policy, [(1,2)], 8, Q1)
    V2 = RLCore.QtoV(Q2)
    GridWorld.printV(gw, V2)
    GridWorld.printActionsFromQ(gw, Q2)
    #RiverSwim.printV(V2)
    #RiverSwim.printActionsFromQ(Q2)
