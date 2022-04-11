from environments.OfficeWorld import officeworld
from training.QLearningCRM import policyEpsilonGreedy, policyRandom, qLearn
from environments import GridWorld
import RLCore
import numpy as np

if __name__ == '__main__':

    of = officeworld(5, 5, [(1,1)], [(4,4)], [(1,3)], 0.9)
    Q = qLearn(of, policyRandom(of.actions), (0,0), 'start')
    Q = np.array(Q)
    Qstart = Q[:,1,:]
    Vstart = RLCore.QtoV(Qstart.tolist())  # type: ignore
    print(Qstart.shape)
    GridWorld.printV(of, Vstart)  # type: ignore
    GridWorld.printActionsFromQ(of, Qstart.tolist())  # type: ignore
    