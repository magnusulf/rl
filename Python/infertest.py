import training
from environments import OfficeWorld, InferredRM
from environments.InferredRM import inferredRM
from training import QLearningCRM, QLearningCPB,ValueIterationRM, JIRP2, JIRP3
import RLCore
import numpy as np

from typing import Set, List
from inferrer import utils, inferrer
import translator
import matplotlib.pyplot as plt

of = OfficeWorld.officeworld(3, 3,
    [],
    [(1,1)], # Office
    [(1,0), (0, 2)], # Coffes
    [], # Decorations
    0.1, # turn probability
    0.95) # discount
s1 = (0, 0)
u1 = 'start'

# Q = JIRP3.qLearn(of, 0.2, [s1], 'start', 100000, 100)
base_mdprm = of
epsilon = 0.2
num_episodes = 50
episode_length = 1000
max_trace_len = 8

initialS = [s1]
initialU = u1

real_regex = '(cc*(o(o|c))|o)*cc*o'

#infer initial hypothesis
X = []
hypothesis = inferredRM(base_mdprm)
hypothesis.inferRewardMachine(X, max_trace_len)
#initialize Q and visitCount
maxQ: float = 1.0/(1.0-hypothesis.discount)
Q = [[[maxQ for _ in hypothesis.actions] for _ in hypothesis.reward_states] for _ in hypothesis.states]
visitCount = [[0 for _ in hypothesis.actions] for _ in hypothesis.states]
#policy
policy = JIRP3.policyEpsilonGreedy(hypothesis, epsilon)

plot_polDiffs = []
plot_Qdiffs = []
plot_iter = []

for episode in range(num_episodes):
    #print("episode "+str(episode))
    Q, new_X = JIRP3.CRM_episode(hypothesis, base_mdprm.rewardTransition, policy, initialS, initialU, Q, visitCount, episode_length)

    # infer new hypothesis
    if len(new_X) > 0:
        
        print("episode "+str(episode))
        X = X + new_X
        prevrewardmachine = hypothesis.dfa.to_regex()
        hypothesis.inferRewardMachine(X, max_trace_len)
        # re-init Q and visitCount
        if (prevrewardmachine != hypothesis.dfa.to_regex()):
            Q = [[[maxQ for _ in hypothesis.actions] for _ in hypothesis.reward_states] for _ in hypothesis.states]
            visitCount = [[0 for _ in hypothesis.actions] for _ in hypothesis.states]


print(hypothesis.dfa.to_regex())
for state in hypothesis.dfa.states:
    _, r = hypothesis.rewardTransition(state.name, ['coffee'])
    print("coffee statename:", state.name, "reward:", r)
    _, r = hypothesis.rewardTransition(state.name, ['office'])
    print("office statename:", state.name, "reward:", r)