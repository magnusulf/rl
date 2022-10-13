from typing import cast

from matplotlib.pyplot import grid
import MDP
import RLCore
import numpy as np
from environments import RiverSwim
import mdprm

class riverswimRM(mdprm.mdprm[int, str, str]):
    def __init__(self, size: int, stayProbability: float, reverseProbability: float, leftReward: float, rightReward: float, discount: float):
        self.size = size
        self.stayProbability = stayProbability
        self.reverseProbability = reverseProbability
        self.leftReward = leftReward
        self.rightReward = rightReward
        states = [x for x in range(size)]
        actions = ['left ', 'right']
        reward_states = ['rightgoal', 'leftgoal']
        mdprm.mdprm.__init__(self, states, actions, reward_states, discount, "Riverswim")

    def rewardTransition(self, u: str, labels: 'list[str]') -> 'tuple[str, float]':
        if (u == 'rightgoal' and 'right' in labels):
            return 'leftgoal', 1
        if (u == 'leftgoal' and 'left' in labels):
            return 'rightgoal', 1
        return u, 0

    def labelingFunction(self, s1: 'int', a: str, s2: 'int') -> 'list[str]':
        ret = []
        if (s1 == 0): ret.append("left")
        if (s1 == self.size-1): ret.append("right")
        return ret

    def baseTransition(self, s1: int, a: str) -> 'list[tuple[int, float]]':
        if (s1 == 0 and a == 'left '):
            return [(0, 1.0)]
        elif (s1 == 0 and a == 'right'):
            return [(0, self.stayProbability + self.reverseProbability), (1, 1.0 - self.stayProbability - self.reverseProbability)]
        elif (s1 == (self.size-1) and a == 'right'):
            return [(s1, self.stayProbability + self.reverseProbability), (s1-1, 1.0 - self.stayProbability - self.reverseProbability)]
        elif (a == 'left '):
            return [(s1-1, 1.0)]
        else:
            return [(s1-1, self.reverseProbability), (s1, self.stayProbability),
            (s1+1, 1.0 - self.reverseProbability - self.stayProbability)]

def idxToAction(action: int) -> str :
    return ['left ', 'right'][action]



def printVs(rs: riverswimRM, Q):
    Qarr = np.array(Q)
    for i in range(len(rs.reward_states)):
        print()
        print(rs.reward_states[i])
        Qstate = Qarr[:,i,:]
        V = RLCore.QtoV(Qstate) # type: ignore
        RiverSwim.printV(V) # type: ignore

def printActions(rs: riverswimRM, Q):
    Qarr = np.array(Q)
    for i in range(len(rs.reward_states)):
        print()
        print(rs.reward_states[i])
        Qstate = Qarr[:,i,:]
        RiverSwim.printActionsFromQ(Qstate) # type: ignore