from multiprocessing.sharedctypes import Value
import GridWorld
import policyIteration
import RiverSwim
import ValueIteration
import QLearning
import UCBLearning
import RLCore
from GridWorld import gridworld
from RiverSwim import riverswim

if __name__ == '__main__':
    #let turnProbability = 0.1
    #let stayProbability = 0.1
    #let livingReward = 0.0

    
    gw = gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.50)
    rs = riverswim(8, 0.55, 0.05, 0.1, 1.0, 0.6)
    
    print("Value iteration")
    #Q = ValueIteration.valueIteration(gw)
    Q1 = ValueIteration.valueIteration(rs)
    V1 = RLCore.QtoV(Q1)
    #GridWorld.printV(gw, V)
    #GridWorld.printActionsFromQ(gw, Q)
    RiverSwim.printV(V1)
    RiverSwim.printActionsFromQ(Q1)

    print("Q learning")
    policy = QLearning.policyEpsilonGreedy(rs, 0.1)
    #Q2 = QLearning.qLearn(gw, policy, gw.starting_state)
    Q2 = QLearning.qLearn(rs, policy, 0)
    V2 = RLCore.QtoV(Q2)
    #GridWorld.printV(gw, V2)
    #GridWorld.printActionsFromQ(gw, Q2)
    RiverSwim.printV(V2)
    RiverSwim.printActionsFromQ(Q2)

    print("Policy iteration")
    #pol, V3 = policyIteration.policyIteration(gw)
    pol, V3 = policyIteration.policyIteration(rs)
    #GridWorld.printV(gw, V3)
    #GridWorld.printActionsFromPolicy(gw, pol)
    RiverSwim.printV(V3)
    RiverSwim.printActionsFromPolicy(pol)