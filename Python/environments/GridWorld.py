import aifc
from typing import Any, cast

from matplotlib.pyplot import grid
import MDP
import RLCore
import numpy as np

class gridworld(MDP.mdp['tuple[int, int]', str]):
    def __init__(self, max_x: int, max_y: int, blocked: 'list[tuple[int, int]]', absorption: 'dict[tuple[int, int], float]', start: 'tuple[int, int]', turnProb: float, discount: float):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)] + [(-1,-1)]
        actions = ['up   ', 'down ', 'left ', 'right']
        self.blocked_states: 'list[tuple[int, int]]' = blocked
        self.absorption_states: 'dict[tuple[int, int], float]' = absorption
        self.starting_state = start
        self.turnProb = turnProb
        MDP.mdp.__init__(self, states, actions, discount, 'gridworld')

    def reward(self, s1, a):
        if (s1 in self.absorption_states):
            return self.absorption_states[s1]
        return 0

    def isTerminal(self, s: 'tuple[int, int]') -> bool:
        # blocked states are deemed as terminal so that Q learning
        # knows that their values are uninteresting
        return s == (-1,-1) or s in self.blocked_states

    def baseTransition(self, s1: 'tuple[int, int]', a: str) -> 'list[tuple[tuple[int, int], float]]':
        # if absorption state
        if (s1 in self.absorption_states):
            return [((-1,-1), 1.0)]

        if (s1 in self.blocked_states):
            return [(s1, 1)]
        
        targets: 'list[tuple[tuple[int, int], float]]' = [((0,0), 1.0)]
        # else try to move
        if (a == 'up   '):
            targets = [
                ((s1[0], s1[1]+1), 1 - (2 * self.turnProb)),
                ((s1[0]-1, s1[1]), self.turnProb),
                ((s1[0]+1, s1[1]), self.turnProb)]
        if (a == 'down '):
            targets = [
                ((s1[0], s1[1]-1), 1 - (2 * self.turnProb)),
                ((s1[0]-1, s1[1]), self.turnProb),
                ((s1[0]+1, s1[1]), self.turnProb)]
        if (a == 'left '):
            targets = [
                ((s1[0]-1, s1[1]), 1 - (2 * self.turnProb)),
                ((s1[0], s1[1]-1), self.turnProb),
                ((s1[0], s1[1]+1), self.turnProb)]
        if (a == 'right'):
            targets = [
                ((s1[0]+1, s1[1]), 1 - (2 * self.turnProb)),
                ((s1[0], s1[1]-1), self.turnProb),
                ((s1[0], s1[1]+1), self.turnProb)]
        # prevent moving into walls
        for i, target in enumerate(targets):
            if (target[0] in self.blocked_states or not self.insideBoundaries(target[0])):
                targets[i] = (s1, targets[i][1])
        return targets

    def insideBoundaries(self, s):
        return (s[0] >= 0 and s[1] >= 0 and s[0] < self.max_x and s[1] < self.max_y)

    def idxToAction(self, action: int) -> str :
        return self.actions[action]

def printV (mdp: gridworld, V: 'list[float]'):
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join(["+{:.2f}".format(V[mdp.stateIdx((x,y))]) for x in range(mdp.max_x)]).replace('+-', '-').replace('nan', 'nan '))


def stateQToString (mdp: gridworld, Q: 'list[list[float]]', state: 'tuple[int, int]') -> str :
    if (state in mdp.blocked_states):
        return "-----"
    elif (state in [x for x in mdp.absorption_states]):
        return "trmnl"
    else:
        idx = mdp.stateIdx(state)
        actionValues = Q[idx]
        aIdx: int = cast(int, np.argmax(actionValues))
        return mdp.idxToAction(aIdx)

def statePolToString (mdp: gridworld, policy: 'list[str]', state: 'tuple[int, int]') -> str :
    if (state in mdp.blocked_states):
        return "-----"
    elif (state in [x for x in mdp.absorption_states]):
        return "trmnl"
    else:
        return policy[mdp.stateIdx(state)]


def printActionsFromQ(mdp: gridworld, Q: 'list[list[float]]') :
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join([stateQToString(mdp, Q, (x, y)) for x in range(mdp.max_x)]))

def printActionsFromPolicy(mdp: gridworld, policy: 'list[str]') :
    for y in range(mdp.max_y-1, -1, -1):
        print(' '.join([statePolToString(mdp, policy, (x, y)) for x in range(mdp.max_x)]))
