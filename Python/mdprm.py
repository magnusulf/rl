import numpy as np
from typing import Any, Callable, Generic, TypeVar

S = TypeVar('S')
A = TypeVar('A')
U = TypeVar('U')

class mdprm(Generic[S, A, U]):
    def __init__(self, states: 'list[S]', actions: 'list[A]', reward_states: 'list[U]', discount: float):
        self.states: 'list[S]' = states
        self.actions: 'list[A]' = actions
        self.reward_states: 'list[U]' = reward_states
        self.discount = discount

    # Due to the implementation of this it is not a true MDP
    # but this one is the easiest to define and can easily be turned
    # to a real transition function or into a stochastic one
    def baseTransition(self, s1: S, a: A) -> 'list[tuple[S, float]]':
        return []

    def rewardTransition(self, u: U, labels: 'list[str]') -> U:
        return None   # type: ignore

    def labelingFunction(self, s1: S, a: A, s2: S) -> 'list[str]':
        return []

    def reward(self, u: U , s1: S, a: A, s2: S) -> float:
        return 0

    def stateIdx(self, s: S) -> int:
        return self.states.index(s)

    def actionIdx(self, a: A) -> int:
        return self.actions.index(a)
    
    def rewardStateIdx(self, u: U) -> int:
        return self.reward_states.index(u)
