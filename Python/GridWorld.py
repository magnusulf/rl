import aifc
from typing import Any, cast

from matplotlib.pyplot import grid
import MDP
import RLCore
import numpy as np

class gridworld(MDP.mdp['tuple[int, int]', str]):
    def __init__(self, max_x: int, max_y: int, blocked: 'list[tuple[int, int]]', absorption: 'dict[tuple[int, int], float]', start: 'tuple[int, int]', discount: float):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)]
        actions = ['up   ', 'down ', 'left ', 'right']
        self.blocked_states: 'list[tuple[int, int]]' = blocked
        self.absorption_states: 'dict[tuple[int, int], float]' = absorption
        self.starting_state = start
        MDP.mdp.__init__(self, states, actions, discount)

    def reward(self, s1, a):
        if (s1 in self.absorption_states):
            return self.absorption_states[s1]
        return 0

    def baseTransition(self, s1: 'tuple[int, int]', a: str) -> 'list[tuple[tuple[int, int], float]]':
        # if absorption state
        if (s1 in self.absorption_states):
            return [(self.starting_state, 1.0)]
        target: 'tuple[int, int]' = 0,0
        # else try to move
        if (a == 'up   '):
            target = (s1[0], s1[1]+1)
        if (a == 'down '):
            target = (s1[0], s1[1]-1)
        if (a == 'left '):
            target = (s1[0]-1, s1[1])
        if (a == 'right'):
            target = (s1[0]+1, s1[1])
        # prevent moving into walls
        if (target in self.blocked_states or not self.insideBoundaries(target) or s1 in self.blocked_states):
            target = s1
        
        return [(target, 1.0)]

    def insideBoundaries(self, s):
        return (s[0] >= 0 and s[1] >= 0 and s[0] < self.max_x and s[1] < self.max_y)

    def idxToAction(self, action: int) -> str :
        return self.actions[action]

def printV (mdp: gridworld, V: 'list[float]'):
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join(["+{:.2f}".format(V[mdp.stateIdx((x,y))]) for x in range(mdp.max_x)]).replace('+-', '-'))


def stateQToString (mdp: gridworld, Q: 'list[list[float]]', state: 'tuple[int, int]') -> str :
    if (state in mdp.blocked_states):
        return "-----"
    elif (state in [x for x in mdp.absorption_states]):
        return "Rward"
    else:
        idx = mdp.stateIdx(state)
        actionValues = Q[idx]
        aIdx: int = cast(int, np.argmax(actionValues))
        return mdp.idxToAction(aIdx)

def statePolToString (mdp: gridworld, policy: 'list[str]', state: 'tuple[int, int]') -> str :
    if (state in mdp.blocked_states):
        return "-----"
    elif (state in [x for x in mdp.absorption_states]):
        return "Rward"
    else:
        return policy[mdp.stateIdx(state)]


def printActionsFromQ(mdp: gridworld, Q: 'list[list[float]]') :
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join([stateQToString(mdp, Q, (x, y)) for x in range(mdp.max_x)]))

def printActionsFromPolicy(mdp: gridworld, policy: 'list[str]') :
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join([statePolToString(mdp, policy, (x, y)) for x in range(mdp.max_x)]))
