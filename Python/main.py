from multiprocessing.sharedctypes import Value
import GridWorld
import ValueIteration
import QLearning
from GridWorld import gridworld


if __name__ == '__main__':
    #let turnProbability = 0.1
    #let stayProbability = 0.1
    #let livingReward = 0.0
    gw = gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.899)
    Q = ValueIteration.valueIteration(gw)
    GridWorld.printV(gw, Q)
    GridWorld.printActions(gw, Q)

    Q2 = QLearning.qLearn(gw, QLearning.policyEpsilonGreedy(gw, 0.05), gw.starting_state)
    GridWorld.printV(gw, Q2)
    GridWorld.printActions(gw, Q2)