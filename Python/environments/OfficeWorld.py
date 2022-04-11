import mdprm 

class officeworld(mdprm.mdprm['tuple[int, int]', str, str]):
    def __init__(self, max_x: int, max_y: int, blocked: 'list[tuple[int, int]]', offices: 'list[tuple[int, int]]', coffee: 'list[tuple[int, int]]', discount: float):
        self.max_x = max_x
        self.max_y = max_y
        states = [(x,y) for x in range(max_x) for y in range(max_y)]
        actions = ['up   ', 'down ', 'left ', 'right']
        reward_states = ['start', 'coffee', 'terminal']
        self.blocked_states: 'list[tuple[int, int]]' = blocked
        self.office_states: 'list[tuple[int, int]]' = offices
        self.coffee_states: 'list[tuple[int, int]]' = coffee
        self.absorption_states = []
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
        return u, 0

    def labelingFunction(self, s1: 'tuple[int, int]', a: str, s2: 'tuple[int, int]') -> 'list[str]':
        ret = []
        if (s2 in self.office_states): ret.append("office")
        if (s2 in self.coffee_states): ret.append("coffee")
        return ret

    def isTerminal(self, u: str):
        return u == 'terminal'

    def baseTransition(self, s1: 'tuple[int, int]', a: str) -> 'list[tuple[tuple[int, int], float]]':
        # if absorption state
        target: 'tuple[int, int]' = (0,0)
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
        if (target in self.blocked_states or not self.insideBoundaries(target)):
            target = s1
        
        return [(target, 1.0)]
    
    def insideBoundaries(self, s):
        return (s[0] >= 0 and s[1] >= 0 and s[0] < self.max_x and s[1] < self.max_y)

    def idxToAction(self, action: int) -> str :
        return self.actions[action]
        