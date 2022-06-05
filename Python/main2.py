from environments import GridWorld
from training import ValueIteration, DeepQL2, QLearning
import RLCore

if __name__ == '__main__':

    #gw = GridWorld.gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.1, 0.50)
    gw = GridWorld.gridworld(2, 2, [], {(1,1) : 1.0}, (0,0), 0.1, 0.90)
    
    print("Value iteration baseline")
    Q1 = ValueIteration.valueIteration(gw)
    V1 = RLCore.QtoV(Q1)
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)

    print("Deep QLearning")
    policy = DeepQL2.policyEpsilonGreedy(gw, 0.1)
    Q2 = DeepQL2.dqLearn(gw, policy, (0,0))
    V2 = RLCore.QtoV(Q2)
    # GridWorld.printV(gw, V2)
    # GridWorld.printActionsFromQ(gw, Q2)

    # print("Q learning")
    # policy = QLearning.policyEpsilonGreedy(gw, 0.1)
    # Q2 = QLearning.qLearn(gw, policy, (0,0))