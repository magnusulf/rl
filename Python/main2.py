from environments import GridWorld
from training import ValueIteration, DeepQL
import RLCore

if __name__ == '__main__':

    gw = GridWorld.gridworld(8, 7, [(0,3), (1,3), (2,3), (3,3), (5,3), (6,3), (7,3)], {(7,0) : 1.0}, (0,6), 0.1, 0.50)
    
    Q1 = ValueIteration.valueIteration(gw)
    V1 = RLCore.QtoV(Q1)
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)