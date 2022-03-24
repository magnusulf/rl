from typing import Any, Callable, TypeVar
import random as rnd

# A base transition function is a transition function that takes a state
# and an action and returns a list of tuples containing the ending states and
# the probability of ending in each of those (should sum to 1).
# Any state not in the list has probability 0 of being the destination

S = TypeVar('S')
A = TypeVar('A')

def getRandomElementWeightedSub (lst: 'list[S]', weigths: 'list[float]', r: float) -> S:
    if (r - weigths[0] > 0):
        return getRandomElementWeightedSub(lst[1:], weigths[1:], r - weigths[0])
    else:
        return lst[0]

def getRandomElementWeighted (lst: 'list[S]', weigths: 'list[float]') -> S :
    return getRandomElementWeightedSub(lst, weigths, rnd.random() * sum(weigths))

def baseTransitionFunctionToNormalTransitionFunction(base: 'Callable[[S, A], list[tuple[S, float]]]') -> 'Callable[[S, A, S], float]':
    def normal(sFrom: S, a: A, sTo: S) -> float:
        pairs = base(sFrom, a) # list of tuples with state and probability
        return sum([x[1] for x in pairs if x[0]== sTo])
    return normal

def baseTransitionFunctionToStochasticTransitionFunction(base: 'Callable[[S, A], list[tuple[S, float]]]') -> 'Callable[[S, A], S]':
    def tf(s1: S, a: A) -> S:
        probs = base(s1, a)
        return getRandomElementWeighted([x[0] for x in probs], [x[1] for x in probs])
    return tf

# Given a state and the calculated Q-table get the value of the state
# Which is the action giving the maximum value
def getStateValue(stateIdx: 'Callable[[S], int]', state: S, Q: 'list[list[float]]' ) -> float:
    return max(Q[stateIdx(state)])

# Given a starting state an action and the end state what is the value
# This is equal to the transition reward plus the discounted (expected) value of the end state
def getActionToStateValue(stateIdx: ('Callable[[S], int]'), actionIdx: 'Callable[[A], int]', R: 'list[list[float]]',
                        discount: float, stateFrom: S, action: A, stateTo: S, Q: 'list[list[float]]') -> float:
    transitionReward: float = R[stateIdx(stateFrom)][actionIdx(action)]
    nextStateValue: float = getStateValue(stateIdx, stateTo, Q)
    return transitionReward + discount * nextStateValue

    