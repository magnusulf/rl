import numpy as np
from typing import Any, Callable, Generic, TypeVar

S = TypeVar('S')
A = TypeVar('A')

class mdp(Generic[S, A]):
    def __init__(self, states: 'list[S]', actions: 'list[A]', discount: float, maxStateIdx: int):
        self.states: 'list[S]' = states
        self.actions: 'list[A]' = actions
        self.discount = discount
        self.maxStateIdx = maxStateIdx

    # Due to the implementation of this it is not a true MDP
    # but this one is the easiest to define and can easily be turned
    # to a real transition function or into a stochastic one
    def baseTransition(self, s1: S, a: A) -> 'list[tuple[S, float]]':
        return []

    def reward(self, s: S, a: A) -> float:
        return 0

    def stateIdx(self, s: S) -> int:
        return self.states.index(s)

    def actionIdx(self, a: A) -> int:
        return self.actions.index(a)
