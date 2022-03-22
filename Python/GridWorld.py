import aifc
from typing import cast

from matplotlib.pyplot import grid
import MDP
import RLCore
import numpy as np

class gridworld(MDP.mdp['tuple[int, int]', str]):
    def __init__(self, max_x: int, max_y: int, blocked: 'list[tuple[int, int]]', absorption: 'dict[tuple[int, int], float]', start: 'tuple[int, int]', discount: float):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)]
        actions = ['up', 'down', 'left', 'right']
        self.blocked_states: 'list[tuple[int, int]]' = blocked
        self.absorption_states: 'dict[tuple[int, int], float]' = absorption
        self.starting_state = start
        MDP.mdp.__init__(self, states, actions, discount, max_x * max_y - 1)

    def reward(self, s1, a):
        if (s1 in self.absorption_states):
            return self.absorption_states[s1]
        return 0

    def transition(self, s1: 'tuple[int, int]', a: str, s2: 'tuple[int, int]'):
        # if absorption state
        if (s1 in self.absorption_states):
            if (s2 == self.starting_state): 
                return 1
            return 0
        target: 'tuple[int, int]' = 0,0
        # else try to move
        if (a == 'up'):
            target = (s1[0], s1[1]+1)
        if (a == 'down'):
            target = (s1[0], s1[1]-1)
        if (a == 'left'):
            target = (s1[0]-1, s1[1])
        if (a == 'right'):
            target = (s1[0]+1, s1[1])
        # prevent moving into walls
        if (target in self.blocked_states or not self.insideBoundaries(target)):
            target = s1
        
        if (target == s2):
            return 1
        return 0

    def insideBoundaries(self, s):
        return (s[0] >= 0 and s[1] >= 0 and s[0] < self.max_x and s[1] < self.max_y)

    def stateIdx(self, s: 'tuple[int, int]'):
        return (s[0]) * (self.max_y) + (s[1])

    def actionIdx(self, a):
        if (a == 'up'):
            return 0
        elif (a == 'down'):
            return 1
        elif (a == 'left'):
            return 2
        elif (a == 'right'):
            return 3

def idxToAction(action: int) -> str :
    if (action == 0): return 'up   '
    if (action == 1): return 'down '
    if (action == 2): return 'left '
    if (action == 3): return 'right'
    return ''

def printV (mdp: gridworld, Q):
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join(["+{:.2f}".format(RLCore.getStateValue(mdp.stateIdx, (x, y), Q)) for x in range(mdp.max_x)]).replace('+-', '-'))


def stateQToString (mdp: gridworld, Q: 'list[list[float]]', state: 'tuple[int, int]') -> str :
    if (state in mdp.blocked_states):
        return "-----"
    elif (state in [x for x in mdp.absorption_states]):
        return "Rward"
    else:
        idx = mdp.stateIdx(state)
        actionValues = Q[idx]
        aIdx: int = cast(int, np.argmax(actionValues))
        return idxToAction(aIdx)


def printActions(mdp: gridworld, Q: 'list[list[float]]') :
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join([stateQToString(mdp, Q, (x, y)) for x in range(mdp.max_x)]))
