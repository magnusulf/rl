from environments import GridWorld, RiverSwim, OfficeWorld
from training import ValueIteration, QLearning
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
    initialStates: 'list[int]' = [1]

    rs = RiverSwim.riverswim(8, 0.55, 0.05, 0.1, 1.0, 0.95)

    print("Value iteration")
    Q1 = ValueIteration.valueIteration(rs)
    V1 = RLCore.QtoV(Q1)
    RiverSwim.printV(V1)
    RiverSwim.printActionsFromQ(Q1)

    # Learning rate comparisons
    # qls1: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateFixed(0.05), QLearning.stateSupplierRandom(initialStates), 'fixed: 0.05', 'red')
    # qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateAvg(), QLearning.stateSupplierRandom(initialStates), 'divide visit #', 'purple')
    # qls3: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateCta(10, 0.6), QLearning.stateSupplierRandom(initialStates), 'c=10, a=0.6', 'green')
    # qls4: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateCta(2, 0.8), QLearning.stateSupplierRandom(initialStates), 'c=2, a=0.8', 'yellow')
    # qls5: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateDivideT(), QLearning.stateSupplierRandom(initialStates), 'divide t', 'brown')
    # qls6: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateK(12), QLearning.stateSupplierRandom(initialStates), 'k-12', 'orange')


    qls1: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.2), QLearning.learningRateK(12), QLearning.stateSupplierRandom(initialStates), 'epsilon greedy', 'red')
    qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyRandom(rs.actions), QLearning.learningRateK(12), QLearning.stateSupplierRandom(initialStates), 'random', 'green')
    qls3: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(rs, 0.0), QLearning.learningRateK(12), QLearning.stateSupplierRandom(initialStates), 'full greedy', 'blue')
    
    #qls2: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyEpsilonGreedy(gw, 0.0), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'full-greedy, k-25, many-state', 'green')
    #qls3: QLearning.qLearnSetting = QLearning.qLearnSetting(QLearning.policyRandom(gw.actions), QLearning.learningRateK(25), QLearning.stateSupplierRandom(initialStates), 'random, k-25, many-states', 'red')

    QLearning.qLearnMany(rs, [qls1, qls2, qls3], 200_000, Q1, 200)