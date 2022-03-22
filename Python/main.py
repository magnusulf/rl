from multiprocessing.sharedctypes import Value
import GridWorld
import ValueIteration
from GridWorld import gridworld


if __name__ == '__main__':
    #let turnProbability = 0.1
    #let stayProbability = 0.1
    #let livingReward = 0.0
    gw = gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.9)
    Q = ValueIteration.valueIteration(gw)
    GridWorld.printV(gw, Q)
    GridWorld.printActions(gw, Q)