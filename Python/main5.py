import training
from environments import OfficeWorld, InferredRM
from training import QLearningCRM, QLearningCPB,ValueIterationRM, JIRP2, JIRP3
import RLCore
import numpy as np

from typing import Set
from inferrer import utils, inferrer
import translator

of = OfficeWorld.officeworld(3, 3,
    [],
    [(1,1)], # Office
    [(1,0), (0, 2)], # Coffes
    [], # Decorations
    0.1, # turn probability
    0.95) # discount
s1 = (0, 0)
u1 = 'start'

# hypothesis = InferredRM.inferredRM(of)
# hypothesis.inferRewardMachine([])

# print(hypothesis.dfa.to_regex()) #
# (cc*(o(o|c))|o)*cc*o
# print([(state, hypothesis.isTerminal(state)) for state in hypothesis.reward_states])
# u1 = hypothesis.dfa._start_state.name
# Q, X = JIRP2.qLearn(hypothesis, QLearningCRM.policyEpsilonGreedy(hypothesis, 0.2), [s1], [u1], 1000)


# pos_examples: Set[str] = set(["".join([hypothesis.lang.fromLabel(e.l) for e in trace]) for trace in X])
# print(pos_examples)
# for trace in X:
#     if len(trace) > 0:
#         print("".join([hypothesis.lang.fromLabel(e.l) for e in trace]))
#         print([trace[-1].r])

Q = JIRP3.qLearn(of, 0.2, [s1], 'start', 100000, 100)