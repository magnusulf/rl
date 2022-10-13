from typing import Tuple
from logic.formula import *
import random as rnd

GMstate = Tuple[formula, formula]
GMPstate = Tuple[GMstate, ...]
ACstate = Tuple[formula, GMPstate]

def toStrGMstate(o: GMstate) -> str:
    return "(" + o[0].toString() + ", " + o[1].toString() + ")"

def toStrGMPstate(o: GMPstate) -> str:
    return "[" + ", ".join([toStrGMstate(x) for x in o]) + "]"

def toStrACstate(o: ACstate) -> str:
    return "<" + o[0].toString() + ", " + toStrGMPstate(o[1]) + ">"

def toStrACstateComplicated(symbols: Set[Symbol], o: ACstate, asc: 'AcceptingSubComponent') -> str:
    return toStrACstate(o) + "\n    " + '\n    '.join([letterToString(symbols, s[0]) + ": " + toStrACstate(s[1]) for s in asc.adjacentStates(o)])

def isSink(acs: ACstate) -> bool:
    if (isinstance(acs[0], formulaFalse)):
        return True
    for gms in acs[1]:
        if (isinstance(gms[0], formulaFalse) or isinstance(gms[1], formulaFalse)):
            return True
    return False

class SimpleComponent():
    def __init__(self, letters: 'set[Letter]', frml: formula, isInitial: bool):
        self.letters = letters
        self.q0 = frml
        self.states = Reach(self.q0, self.letters)
        self.isInitial = isInitial

    def transition(self, s1: formula, v: Letter) -> formula:
        return s1.af(v)

    def isAccepting(self, s1: formula) -> bool:
        if (self.isInitial):
            return False
        return isinstance(s1, formulaTrue)

    def epsilonTransition(self, s1: formula, gList: 'list[formula]') -> 'Tuple[ACstate, AcceptingSubComponent]':
        asc: AcceptingSubComponent = AcceptingSubComponent(self.letters, self.q0, gList)
        ac: ACstate = (s1.gSubstitute(gList), asc.gMonitorProduct.q0)

        return ac, asc

    def __eq__(self, other):
        if not isinstance(other, SimpleComponent): return False
        if (self.letters != other.letters): return False
        if (self.q0 == other.q0): return False
        if (self.isInitial == other.isInitial): return False
        return True

    def __hash__(self):
        return self.letters._hash() + self.q0.__hash__()

class GMonitor():
    def __init__(self, letters: 'set[Letter]', frml: formula):
        self.letters = letters
        self.src = frml
        self.q0: GMstate = (frml, formulaTrue())
        self.states = ReachG(self.src, self.letters)

    def transition(self, s1: GMstate, v: Letter) -> GMstate:
        return transitionG(self.src, s1, v)

    def isAccepting1(self, s1: GMstate, v: Letter, s2: GMstate) -> bool:
        return isinstance(s1[0].af(v), formulaTrue)

    def isAccepting2(self, s2: GMstate) -> bool:
        return isinstance(s2[1], formulaTrue)

    def __eq__(self, other):
        if not isinstance(other, GMonitor): return False
        if (self.letters != other.letters): return False
        if (self.src == other.src): return False
        return True

    def __hash__(self):
        return self.letters._hash() + self.src.__hash__()

def transitionG(src: formula, s1: GMstate, v: Letter) -> GMstate:
    if isinstance(s1[0].af(v), formulaTrue):
        return (formulaAnd.of(s1[1].af(v), src), formulaTrue())
    return (s1[0].af(v), formulaAnd.of(s1[1].af(v), src))
def ReachG(src: 'formula', letters: 'set[Letter]') -> 'list[GMstate]':
    ret = []
    A = [(src, formulaTrue())]
    B = []
    #monitor = GMonitor(letters, src)
    while True:
        for fml in A:
            for letter in letters:
                fml2 = transitionG(src, fml, letter)
                if (fml2 not in A and fml2 not in B and fml2 not in ret):
                    B.append(fml2)
        ret.extend(A)
        A = B
        B = []
        if (len(A) == 0): break

    return ret


# The contained formulas should not be formulaGlobal that is implied
class GMonitorProduct():
    def __init__(self, letters: 'set[Letter]', gList: 'list[formula]'):
        self.letters = letters
        self.gList: List[formula] = gList
        self.monitors = [self.Ui(frml) for frml in self.gList]
        self.q0: GMPstate = tuple([m.q0 for m in self.monitors])
        self.states: 'list[GMPstate]' = GMonitorProduct.calcStates(self.monitors)
        

    def Ui(self, G: formula) -> GMonitor:
        frml = G.gSubstitute(self.gList)
        return GMonitor(self.letters, frml)

    @staticmethod
    def calcStates(monitors: 'list[GMonitor]') -> 'list[GMPstate]':
        return GMonitorProduct.appendStates(monitors, [()])

    @staticmethod
    def appendStates(remainingMonitors: 'list[GMonitor]', states: 'list[GMPstate]') -> 'list[GMPstate]':
        if (len(remainingMonitors) == 0):
            return states
        monitor: GMonitor = remainingMonitors[0]

        ret: 'list[GMPstate]' = []
        for state in states:
            for gState in monitor.states:
                newState: GMPstate = state + (gState,)
                ret.append(newState)
        return GMonitorProduct.appendStates(remainingMonitors[1:], ret)

    def transition(self, s1: GMPstate, v: Letter) -> GMPstate:
        if (len(s1) != len(self.monitors)):
            raise Exception("Non-match")

        return tuple([self.monitors[i].transition(s1[i], v) for i in range(len(s1))])

    def acceptingGlist1(self, s1: GMPstate, v: Letter, s2: GMPstate) -> List[formula]:
        if (len(s1) != len(s2)):
            raise Exception("Non-match 1")
        if (len(s1) != len(self.monitors)):
            raise Exception("Non-match 2")

        ret = []

        for i in range(len(self.monitors)):
            if self.monitors[i].isAccepting1(s1[i], v, s2[i]): 
                ret.append(self.gList[i])
        return ret

    def acceptingGlist2(self,  s2: GMPstate) -> List[formula]:
        ret = []

        for i in range(len(self.monitors)):
            if self.monitors[i].isAccepting2(s2[i]): 
                ret.append(self.gList[i])
        return ret

    def __eq__(self, other):
        if not isinstance(other, GMonitorProduct): return False
        if (self.letters != other.letters): return False
        if (self.gList == other.gList): return False
        return True

    def __hash__(self):
        return self.letters._hash()

class AcceptingSubComponent():
    def __init__(self, letters: 'set[Letter]', initialFormula: formula, gList: 'list[formula]'):
        self.letters = letters
        self.simpleComponent = SimpleComponent(letters, initialFormula, False)
        self.gMonitorProduct = GMonitorProduct(letters, gList)
        #simpleStates = list(set([a.gSubstitute(gList).nextReduce() for a in self.simpleComponent.states]))
        simpleStates = [initialFormula.gSubstitute(gList)]
        self.states: List[ACstate] = [(a,b) for a in simpleStates for b in self.gMonitorProduct.states]
        self.q0: ACstate = (initialFormula.gSubstitute(gList), self.gMonitorProduct.q0)

    def transition(self, s1: ACstate, v: Letter) -> ACstate:
        s2a = self.simpleComponent.transition(s1[0], v)
        s2b = self.gMonitorProduct.transition(s1[1], v)
        return (s2a, s2b)

    def acceptingGlist1(self, s1: ACstate, v: Letter, s2: ACstate) -> List[formula]:
        if (not self.simpleComponent.isAccepting(s1[0])): return []
        if (not self.simpleComponent.isAccepting(s2[0])): return []
        return self.gMonitorProduct.acceptingGlist1(s1[1], v, s2[1])

    def acceptingGlist2(self, s2: ACstate) -> List[formula]:
        if (not self.simpleComponent.isAccepting(s2[0])): return []
        return self.gMonitorProduct.acceptingGlist2(s2[1])
        
    def acceptingConditions(self) -> List[formula]:
        return self.gMonitorProduct.gList

    def getF(self, fml: formula) -> 'set[ACstate]':
        return {x for x in self.states if fml in self.acceptingGlist2(x)}

    def stateIndex(self, acs: ACstate):
        return self.states.index(acs)

    def adjacentStates(self, s1: ACstate) -> List[Tuple[Letter, ACstate]]:
        ret: List[Tuple[Letter, ACstate]] = []
        for v in self.letters:
            s2 = self.transition(s1, v)
            if (s2 in ret): continue
            ret.append((v, s2))
        return ret

    def initialiseA(self) -> Set[ACstate]:
        ret: 'set[ACstate]' = set()
        for c in self.acceptingConditions():
            fj = self.getF(c)
            ret = ret.union(fj)
        return ret

    def randomQ(self):
        q = rnd.choice(self.states)
        while (isSink(q)):
            q = rnd.choice(self.states)
        return q

    def Acc(self, q: ACstate, F: Set[ACstate]) -> Set[ACstate]:
        for condition in self.acceptingConditions():
            fj = self.getF(condition)
            if (not q in fj): continue
            if (fj != F):
                return F - fj
            else:
                ret: 'set[ACstate]' = set()
                for c in self.acceptingConditions():
                    ret = ret.union(self.getF(c))
                ret = ret - fj
                return ret
        return F
