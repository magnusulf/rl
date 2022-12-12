from typing import Set, List

from matplotlib.font_manager import X11FontDirectories
import mdprm
#from inferrer.automaton.state import State
import translator
from inferrer import utils, inferrer, automaton

class inferredRM(mdprm.mdprm):
    def __init__(self, base_model: mdprm.mdprm):
        self.base_model = base_model
        mdprm.mdprm.__init__(self, base_model.states, base_model.actions, [], base_model.discount, base_model.desc)

    def inferRewardMachine(self, X, max_trace_len=6):
        self.lang = translator.language()

        
        for trace in X:
            for experience in trace:
                self.lang.addLabel(experience.l)

        # transform to pos/neg examples from traces

        pos_examples: Set[str] = set([]) 
        neg_examples: Set[str] = set([])
        for trace in X:
            example = ("".join([self.lang.fromLabel(e.l) for e in trace]))
            if len(example) > 0 and len(example) <= max_trace_len:
                reward = (trace[-1].r)
                if reward == 0:
                    neg_examples.add(example)
                elif reward == 1:
                    pos_examples.add(example)
        
        if len(pos_examples) + len(neg_examples) == 0:
            pos_examples: Set[str] = set(["Ø"])

        alphabet = utils.determine_alphabet(pos_examples.union(neg_examples))

        learner = inferrer.Learner(alphabet=alphabet,
            pos_examples=pos_examples,
            neg_examples=neg_examples,
            algorithm='rpni')
        
        

        #print("Inferring new reward machine from: ", pos_examples, neg_examples)
        dfa = learner.learn_grammar()
        self.dfa: automaton.DFA = dfa
        #print("inferred reward machine " + self.dfa.to_regex() + "\n")

        self.reward_states = [state.name for state in self.dfa.states]
        self.initialState = self.dfa._start_state.name
        self.terminalStates: list[str] = [state.name for state in self.dfa.accept_states]
    
    def dfaStateFromName(self, u: str) -> automaton.State:
        for state in self.dfa.states:
            if state.name == u:
                return state

    def rewardTransition(self, u, labels: 'list[str]') -> 'tuple[str, float]':
        letter = self.lang.fromLabel(labels)
        q = self.dfaStateFromName(u)
        if not self.dfa.transition_exists(q, letter):
            return q.name, 0
        q = self.dfa.transition(q, letter)
        if q in self.dfa.accept_states:
            return q.name, 1
        else:
            return q.name, 0

    def isTerminal(self, u):
        return (u in self.terminalStates)

    def labelingFunction(self, s1, a, s2):
        return self.base_model.labelingFunction(s1, a, s2)

    def baseTransition(self, s1, a) -> 'list[tuple[any, float]]':
        return self.base_model.baseTransition(s1, a)

    def isBlocked(self, s):
        return self.base_model.isBlocked(s)

    def insideBoundaries(self, s):
        return self.base_model.insideBoundaries(s)

    def idxToAction(self, action: int) -> str :
        return self.base_model.idxToAction(action)