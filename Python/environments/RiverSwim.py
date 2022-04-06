from typing import cast

from matplotlib.pyplot import grid
import MDP
import RLCore
import numpy as np

class riverswim(MDP.mdp[int, str]):
    def __init__(self, size: int, stayProbability: float, reverseProbability: float, leftReward: float, rightReward: float, discount: float):
        self.size = size
        self.stayProbability = stayProbability
        self.reverseProbability = reverseProbability
        self.leftReward = leftReward
        self.rightReward = rightReward
        states = [x for x in range(size)]
        actions = ['left ', 'right']
        MDP.mdp.__init__(self, states, actions, discount)

    def reward(self, s1, a) -> float:
        if (s1 == 0):
            return self.leftReward
        elif (s1 == (self.size-1)):
            return self.rightReward
        return 0

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


def printV(V: 'list[float]') :
    print(' '.join(["+{:.2f}".format(val) for val in V]).replace('+-', '-'))


def stateQToString(Q: 'list[list[float]]', state: int) -> str :
    actionIdx: int = cast(int, np.argmax(Q[state]))
    return idxToAction(actionIdx)


def printActionsFromQ(Q: 'list[list[float]]'):
    print(' '.join([stateQToString(Q, x) for x in range(len(Q))]))

def printActionsFromPolicy(policy: 'list[str]'):
    print(' '.join(policy))
