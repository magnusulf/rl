from environments import GridWorld
import RLCore
import mdprm 
import numpy as np

class officeworld(mdprm.mdprm['tuple[int, int]', str, str]):
    def __init__(self, max_x: int, max_y: int, blocked: 'list[tuple[int, int]]', 
        offices: 'list[tuple[int, int]]', coffee: 'list[tuple[int, int]]', 
        decorations: 'list[tuple[int, int]]', turnProbability: float, discount: float):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)]
        actions = ['up   ', 'down ', 'left ', 'right']
        self.turnProb = turnProbability
        reward_states = ['start', 'coffee', 'terminal']
        self.blocked_states: 'list[tuple[int, int]]' = blocked
        self.office_states: 'list[tuple[int, int]]' = offices
        self.coffee_states: 'list[tuple[int, int]]' = coffee
        self.decoration_states: 'list[tuple[int, int]]' = decorations
        self.absorption_states = offices + decorations # Only used for printing via GridWorld functions
        mdprm.mdprm.__init__(self, states, actions, reward_states, discount)

    # def reward(self, u: str, s1: 'tuple[int, int]', a: str, s2: 'tuple[int, int]') -> float:
    #     if (u == 'coffee' and s2 in self.office_states):
    #         return 1
    #     return 0

    def rewardTransition(self, u: str, labels: 'list[str]') -> 'tuple[str, float]':
        if (u == 'start' and 'coffee' in labels):
            return 'coffee', 0
        if (u == 'coffee' and 'office' in labels):
            return 'terminal', 1
        if ('decoration' in labels):
            return 'terminal', 0
        return u, 0

    def labelingFunction(self, s1: 'tuple[int, int]', a: str, s2: 'tuple[int, int]') -> 'list[str]':
        ret = []
        if (s2 in self.decoration_states): ret.append("decoration")
        if (s2 in self.office_states): ret.append("office")
        if (s2 in self.coffee_states or s1 in self.coffee_states): ret.append("coffee")
        return ret

    def isTerminal(self, u: str):
        return u == 'terminal'

    def baseTransition(self, s1: 'tuple[int, int]', a: str) -> 'list[tuple[tuple[int, int], float]]':
        # if absorption state
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

def printVs(of: officeworld, Q):
    Qarr = np.array(Q)
    for i in range(len(of.reward_states)):
        if (of.isTerminal(of.reward_states[i])):
            continue
        print()
        print(of.reward_states[i])
        Qstate = Qarr[:,i,:]
        V = RLCore.QtoV(Qstate) # type: ignore
        GridWorld.printV(of, V) # type: ignore

def printActions(of: officeworld, Q):
    Qarr = np.array(Q)
    for i in range(len(of.reward_states)):
        if (of.isTerminal(of.reward_states[i])):
            continue
        print()
        print(of.reward_states[i])
        Qstate = Qarr[:,i,:]
        GridWorld.printActionsFromQ(of, Qstate) # type: ignore
