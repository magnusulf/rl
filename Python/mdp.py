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

    def transition(self, s1: S, a: A, s2: S) -> float:
        return 0

    def reward(self, s: S, a: A) -> float:
        return 0

    def stateIdx(self, s: S) -> int:
        return 0

    def actionIdx(self, a: A) -> int:
        return 0
