import numpy as np

class mdp:
    def __init__(self, states, actions, discount):
        self.states = states
        self.actions = actions
        self.discount = discount

    def transition(self, s1, a, s2):
        return 0

    def reward(self, s, a):
        return 0

class gridworld(mdp):
    def __init__(self, max_x, max_y, blocked, absorption, start, discount):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)]
        actions = ['up', 'down', 'left', 'right']
        self.blocked_states = blocked
        self.absorption_states = absorption
        self.starting_state = start
        mdp.__init__(self, states, actions, discount)

    def reward(self, s1, a):
        if (s1 in self.absorption_states):
            return self.absorption_states[s1]
        return 0

    def transition(self, s1, a, s2):
        # if absorption state
        if (s1 in self.absorption_states):
            if (s2 == self.starting_state): 
                return 1
            return 0
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

if __name__ == '__main__':
    gw = gridworld(3, 3, [(0,1), (2,1)], {(2,2) : 1}, (0,0), 0.9)
    [print(s, gw.transition(s, 'left', (0,0))) for s in gw.states]