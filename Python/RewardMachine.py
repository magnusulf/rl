from environments.OfficeWorld import officeworld
from training.QLearningCPB import qLearn
from training.QLearningCPB import policyEpsilonGreedy
from environments import GridWorld
import RLCore
import numpy as np

if __name__ == '__main__':

    of = officeworld(3, 3, [], [(2,2)], [(0,2)], 0.9)
    Q = qLearn(of, policyEpsilonGreedy(of, 0.1), (0,0), 'start')
    print(Q)
    Q = np.array(Q)
    Qstart = Q[:,1,:]
    Vstart = RLCore.QtoV(Qstart.tolist())  # type: ignore
    print(Qstart.shape)
    GridWorld.printV(of, Vstart)  # type: ignore
    GridWorld.printActionsFromQ(of, Qstart.tolist())  # type: ignore
    