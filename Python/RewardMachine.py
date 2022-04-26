from environments.OfficeWorld import officeworld
from training.QLearningCRM import policyEpsilonGreedy, policyRandom, qLearn
from environments import GridWorld
import RLCore
import numpy as np

if __name__ == '__main__':

    of = officeworld(5, 5, [(1,1)], [(4,4)], [(1,3)], [(2,2)], 0.1, 0.9)
    Q = qLearn(of, policyRandom(of.actions), (0,0), 'start')
    Q = np.array(Q)

    Qstart = Q[:,0,:]
    Vstart = RLCore.QtoV(Qstart.tolist())  # type: ignore
    GridWorld.printV(of, Vstart)  # type: ignore
    GridWorld.printActionsFromQ(of, Qstart.tolist())  # type: ignore

    print("")
    print("SWITCH")

    Q1 = Q[:,1,:]
    V1 = RLCore.QtoV(Q1.tolist())  # type: ignore
    GridWorld.printV(of, V1)  # type: ignore
    GridWorld.printActionsFromQ(of, Q1.tolist())  # type: ignore
    