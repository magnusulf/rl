import numpy as np
from typing import Any, Callable, Generic, TypeVar
import RLCore

S = TypeVar('S')
A = TypeVar('A')

class mdp(Generic[S, A]):
    def __init__(self, states: 'list[S]', actions: 'list[A]', discount: float, desc: str):
        self.states: 'list[S]' = states
        self.actions: 'list[A]' = actions
        self.discount = discount
        self.desc = desc

    # Due to the implementation of this it is not a true MDP
    # but this one is the easiest to define and can easily be turned
    # to a real transition function or into a stochastic one
    def baseTransition(self, s1: S, a: A) -> 'list[tuple[S, float]]':
        return []

    def reward(self, s: S, a: A) -> float:
        return 0

    def isTerminal(self, s: S) -> bool:
        return False

    def stateIdx(self, s: S) -> int:
        return self.states.index(s)

    def actionIdx(self, a: A) -> int:
        return self.actions.index(a)

    def labelingFunction(self, s1: S, a: A, s2: S) -> 'frozenset[str]':
        return frozenset()


def getRewardMatrix(mdp: mdp) -> 'list[list[float]]':
    return [[mdp.reward(s, a) for a in mdp.actions] for s in mdp.states]

#  Calculates a matrix that stores the transition probabilities
# We store it so it needs not be calculated often
def getTransitionMatrix(mdp: mdp) -> 'list[list[list[float]]]':
    transitionF = RLCore.baseTransitionFunctionToNormalTransitionFunction(mdp.baseTransition)
    return [[[transitionF(s1, a, s2) for s2 in mdp.states] for a in mdp.actions] for s1 in mdp.states]