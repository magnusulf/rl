import training
from environments import OfficeWorld
from training import QLearningCRM, QLearningCPB,ValueIterationRM, JIRP, JIRP2
import RLCore
import numpy as np

from typing import Set
from inferrer import utils, inferrer
import translator
# pos_examples: Set[str] = set(['aab', 'ab'])
# neg_examples: Set[str] = set(['bcd'])
# alphabet = utils.determine_alphabet(pos_examples.union(neg_examples))

# for e in alphabet:
#     print(e)

if 'c' in ['a', 'b']:
    print("a in list")

of = OfficeWorld.officeworld(3, 3,
    [],
    [(1,1)], # Office
    [(2,0), (0, 2)], # Coffes
    [], # Decorations
    0.1, # turn probability
    0.95) # discount
s1 = (0, 0)
u1 = 'start'

# print("Value iteration RM")
# Qvi = ValueIterationRM.valueIteration(of)
# Qvi = np.array(Qvi)

#print(Qvi)

#Q = JIRP.qLearn(of, JIRP.policyEpsilonGreedy(of, 0.2), s1, u1, 1_000)
iterations = 100
#Q,_,_,_ = QLearningCRM.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), [s1], [u1], iterations, Qvi, 1)
Q, X = JIRP2.qLearn(of, QLearningCRM.policyEpsilonGreedy(of, 0.2), [s1], [u1], iterations)

lang = translator.language()

for trace in X:
    for experience in trace:
        lang.addLabel(experience.l)

#print(lang.dict)

pos_examples: Set[str] = set(["".join([lang.fromLabel(e.l) for e in trace]) for trace in X]) 
neg_examples: Set[str] = set(["o"])
alphabet = utils.determine_alphabet(pos_examples.union(neg_examples))

for e in alphabet:
    print(e)

learner = inferrer.Learner(alphabet=alphabet,
    pos_examples=pos_examples,
    neg_examples=neg_examples,
    algorithm='rpni')

dfa = learner.learn_grammar()
print(dfa.to_regex())
print(pos_examples)

#OfficeWorld.printVs(of, Q)
OfficeWorld.printActions(of, Q)

# print(X[1])
# print(X[1][0])
# print("".join([lang.fromLabel(e.l) for e in X[1]]))