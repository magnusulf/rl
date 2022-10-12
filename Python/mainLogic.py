from logic import lcrl
from logic.formula import *
from logic.gMonitor import *
from environments import GridWorld,  RiverSwim, OfficeWorld
from training import QLearningCRM, ValueIterationRM

if __name__ == '__main__':
    symbols = {'coffee', 'office', 'decoration'}
    letters = symbols2letters(symbols)
    coffe = formulaSymbol('coffee')
    office = formulaSymbol('office')
    decoration = formulaSymbol('decoration')
    tt = formulaTrue()
    ff = formulaFalse()
    
    g1: formula = formulaNegationSymbol('decoration')
    g2: formula = formulaFuture(formulaAnd.of(coffe, formulaFuture(office)))

    phi: formula = formulaGlobal(g2)

    asc = AcceptingSubComponent(letters, phi, [g2])

    print("ASC states")
    for q in asc.states:
        if (isSink(q)): continue
        print(toStrACstate(q))

    of = OfficeWorld.officeworld(11, 11,
        [(3,0), (3,2), (7,0), (7,2), (0,3), (2,3), (2,7), (0,7), (8, 3), (8, 7), (10, 3), (10, 7), (3, 8), (3, 10), (7, 8), (7, 10),
        (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (3, 4), (3, 5), (3, 6), (3, 7), (7, 4), (7, 5), (7, 6), (7, 7), (4, 7), (6, 7)],
        [(5,5)], # Office
        [(4,8), (4, 2)], # Coffes
        [(1, 5), (1, 9), (5,1), (5, 9), (9, 1), (9, 5), (9,9)], # Decorations
        0.1, # turn probability
        0.95) # discount
    ofs = OfficeWorld.officeworldsimple(of)


    s1 = (1, 2)
    u1 = 'start'
    sList = [(1, 1), (8, 1), (8, 9)]

    # print("Value iteration")
    # realQ = ValueIterationRM.valueIteration(of)
    # (Q, plot_diffs, plot_iter, cover_time) = QLearningCRM.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.1), [s1], [u1], 2_000_000, realQ, 1)

    Qltl = lcrl.lcrlLearn(ofs, asc, lcrl.policyEpsilonGreedy(ofs, asc, 0.1), sList, 8, 2_000_000)

    # print("Optimal actions")
    # OfficeWorld.printActions(of, realQ)

    # print("Estimated actions")
    # OfficeWorld.printActions(of, Q)

    print("logic actions")
    OfficeWorld.printActionsSimple(ofs, asc, Qltl)

    # print("real Q")
    # OfficeWorld.printVs(of, realQ)

    # print("estm Q")
    # OfficeWorld.printVs(of, Q)

    print("logic Q")
    OfficeWorld.printVsSimple(ofs, asc, Qltl)