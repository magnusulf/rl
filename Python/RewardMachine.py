from environments.OfficeWorld import officeworld
from training import QLearningCRM, QLearningCPB
from environments import GridWorld
import RLCore
import numpy as np

if __name__ == '__main__':

    of = officeworld(5, 5, [(1,1)], [(4,4)], [(1,3)], [(2,2)], 0.1, 0.89)
    Q = QLearningCRM.qLearn(of, QLearningCRM.policyRandom(of.actions), (0,0), 'start')
    Q = np.array(Q)

    Qstart = Q[:,0,:]
    Vstart = RLCore.QtoV(Qstart.tolist())  # type: ignore
    GridWorld.printV(of, Vstart)  # type: ignore
    GridWorld.printActionsFromQ(of, Qstart.tolist())  # type: ignore

    print("")
    print("SWITCH")

    Qcoffe = Q[:,1,:]
    Vcoffe = RLCore.QtoV(Qcoffe.tolist())  # type: ignore
    GridWorld.printV(of, Vcoffe)  # type: ignore
    GridWorld.printActionsFromQ(of, Qcoffe.tolist())  # type: ignore

    print("")
    print("CPB")

    Q = QLearningCPB.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.1), (0,0), 'start')
    Q = np.array(Q)

    Qstart = Q[:,0,:]
    Vstart = RLCore.QtoV(Qstart.tolist())  # type: ignore
    GridWorld.printV(of, Vstart)  # type: ignore
    GridWorld.printActionsFromQ(of, Qstart.tolist())  # type: ignore

    print("")
    print("SWITCH")

    Qcoffe = Q[:,1,:]
    Vcoffe = RLCore.QtoV(Qcoffe.tolist())  # type: ignore
    GridWorld.printV(of, Vcoffe)  # type: ignore
    GridWorld.printActionsFromQ(of, Qcoffe.tolist())  # type: ignore
    