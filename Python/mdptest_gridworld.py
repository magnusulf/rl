from environments import GridWorld,  RiverSwim, OfficeWorld
from training import ValueIteration, policyIteration, QLearning, UCBLearning
import RLCore

if __name__ == '__main__':
    #let turnProbability = 0.1
    #let stayProbability = 0.1
    #let livingReward = 0.0

    # This is the office world we later use with reward machines.
    # For the sake of similarity we use a gridworld that is identical
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
    {(5, 5): 1, (1, 5) : 0, (1, 9) : 0, (5,1) : 0, (5, 9) : 0, (9, 1) : 0, (9, 5) : 0, (9,9) : 0},
    s1, of.turnProb, of.discount)
    initialStates: 'list[tuple[int, int]]' = [s1, (10,0), (10, 10), (0, 10)]

    rs = RiverSwim.riverswim(4, 0.55, 0.05, 0.1, 1.0, 0.8)

    print("Value iteration")
    Q1 = ValueIteration.valueIteration(gw)
    V1 = RLCore.QtoV(Q1)
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)

    print("Value iteration")
    Q1 = ValueIteration.valueIteration(gw)
    V1 = RLCore.QtoV(Q1)
    GridWorld.printV(gw, V1)
    GridWorld.printActionsFromQ(gw, Q1)

    # Learning rates
    # qls1: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateFixed(0.05), QLearning.stateSupplierRandom(initialStates), 'fixed: 0.05', 'red')
    # qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateCta(8, 0.6), QLearning.stateSupplierRandom(initialStates), 'c=8, a=0.6', 'blue')
    # qls3: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateCta(10, 0.53), QLearning.stateSupplierRandom(initialStates), 'c=10, a=0.53', 'green')
    # qls4: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateCta(1, 0.5), QLearning.stateSupplierRandom(initialStates), 'c=1, a=0.5', 'yellow')
    # qls5: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'k-25', 'orange')
    # qls6: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateAvg(), QLearning.stateSupplierRandom(initialStates), 'divide visit#', 'purple')
    # qls7: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateDivideT(), QLearning.stateSupplierRandom(initialStates), 'divide t', 'brown')
    
    # Policies
    # qls1: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'epsilon greedy', 'red')
    # qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyRandom(gw.actions), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'random', 'green')
    # qls3: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.0), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'full greedy', 'blue')
    
    # initial states
    qls1: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'several states', 'red')
    qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.2), QLearning.learningRateK(25), QLearning.stateSupplierFixed(initialStates), 'single state', 'blue') 

    QLearning.qLearnMany(gw, [qls1, qls2], 1_500_000, Q1, 50)

    # print("\nQ learning (e-greedy) (k-plus learning rate)")
    

    # policy = QLearning.policyEpsilonGreedy(gw, 0.2)
    # #Q2 = QLearning.qLearn(gw, policy, gw.starting_state) , (5, 2), (9,2), (1, 10), (9, 10)
    # Q2 = QLearning.qLearn(gw, policy, [(1,2)], 8, Q1)
    # V2 = RLCore.QtoV(Q2)
    # GridWorld.printV(gw, V2)
    # GridWorld.printActionsFromQ(gw, Q2)
    # #RiverSwim.printV(V2)
    # #RiverSwim.printActionsFromQ(Q2)
