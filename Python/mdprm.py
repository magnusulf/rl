import numpy as np
from typing import Any, Callable, Generic, TypeVar
import RLCore



S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')

class mdprm(Generic[S, A, U]):
    def __init__(self, states: 'list[S]', actions: 'list[A]', reward_states: 'list[U]', discount: float, desc: str):
        self.states: 'list[S]' = states
        self.actions: 'list[A]' = actions
        self.reward_states: 'list[U]' = reward_states
        self.discount = discount
        self.desc = desc

    # Due to the implementation of this it is not a true MDP
    # but this one is the easiest to define and can easily be turned
    # to a real transition function or into a stochastic one
    def baseTransition(self, s1: S, a: A) -> 'list[tuple[S, float]]':
        return []

    def rewardTransition(self, u: U, labels: 'list[str]') -> 'tuple[U, float]':
        return None   # type: ignore

    def labelingFunction(self, s1: S, a: A, s2: S) -> 'list[str]':
        return []

    # def reward(self, u: U , s1: S, a: A, s2: S) -> float:
    #     return 0

    def stateIdx(self, s: S) -> int:
        return self.states.index(s)

    def actionIdx(self, a: A) -> int:
        return self.actions.index(a)
    
    def rewardStateIdx(self, u: U) -> int:
        return self.reward_states.index(u)

    def fullStateIdx(self, fs: 'tuple[U, S]') -> int:
        return self.rewardStateIdx(fs[0]) * len(self.states) + self.stateIdx(fs[1])

    def isTerminal(self, u: U) -> bool:
        return False

    def isBlocked(self, s: S) -> bool:
        return False

def getRewardMatrix(mdp: mdprm) -> 'list[list[list[list[float]]]]':
    return [[[[mdp.rewardTransition(u, mdp.labelingFunction(s1, a, s2))[1] for u in mdp.reward_states] for s2 in mdp.states] for a in mdp.actions] for s1 in mdp.states]
    

#  Calculates a matrix that stores the transition probabilities
# We store it so it needs not be calculated often
def getTransitionMatrix(mdp: mdprm[S, A, U]) -> 'list[list[list[list[list[float]]]]]':
    transitionF = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    def prob(s1: S, u1: U, a: A, s2: S, u2: U) -> float:
        labels = mdp.labelingFunction(s1, a, s2)
        nextU = mdp.rewardTransition(u1, labels)[0]
        if (nextU != u2):
            return 0
        return transitionF(s1, a, s2)

    return [[[[[prob(s1,u1,a,s2,u2) for u2 in mdp.reward_states] for s2 in mdp.states] for a in mdp.actions] for u1 in mdp.reward_states] for s1 in mdp.states]