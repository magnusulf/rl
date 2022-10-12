
import itertools
from typing import Set, List, FrozenSet

from numpy import isin

Symbol = str
Letter = FrozenSet[Symbol]
Word = List[Letter]

# Converts a list of all symbols (size n) to a list of
# all possible letters (size 2^n)
def symbols2letters(symbols2: 'set[Symbol]') -> 'set[Letter]':
    symbols = symbols2.copy()
    if (len(symbols) == 0): 
        l: Letter = frozenset()
        return {l}
    
    symbol = symbols.pop()
    lettersWithoutSymbol: Set[Letter] = symbols2letters(symbols)
    lettersWithSymbol = {letter.union({symbol}) for letter in lettersWithoutSymbol}

    return lettersWithSymbol.union(lettersWithoutSymbol)

def letterToString(symbols: 'set[Symbol]', v: Letter) -> str:
    ret = ""
    for s in symbols:
        if s in v:
            ret += s
        else:
            ret += "".join([x + u'\u0304' for x in s])
    return ret

def Reach(src: 'formula', symbols: 'set[Letter]') -> 'list[formula]':
    ret = []
    A = [src]
    B = []
    i = 0
    while True:
        i+=1
        for fml in A:
            for symbol in symbols:
                fml2 = fml.af(symbol).reduce()
                if (fml2 not in A and fml2 not in B and fml2 not in ret):
                    B.append(fml2)
        ret.extend(A)
        A = B
        B = []
        if (len(A) == 0): break

    return ret

def gFormulaeCombinations(src: 'formula') -> 'list[list[formula]]':
    gFormulas: 'list[formula]' = src.gSubformulae()

    return gFormulaeCombinationsSub(gFormulas, [[]])

def gFormulaeCombinationsSub(remaining: 'list[formula]', result: 'list[list[formula]]') -> 'list[list[formula]]':
    if (len(remaining) == 0):
        return result
    
    add: formula = remaining[0]
    remaining = remaining[1:]
    resultWith: 'list[list[formula]]' = [[add] + x for x in result]
    return gFormulaeCombinationsSub(remaining, result) + gFormulaeCombinationsSub(remaining, resultWith)


class formula():
    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        raise Exception('Not implemented')

    def evaluateTemp(self, word: Word) -> bool:
        raise Exception('Not implemented')

    def af(self, letter: Letter) -> 'formula':
        raise Exception('Not implemented')

    # The contained formulas should not be formulaGlobal that is implied
    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        raise Exception('Not implemented')

    def nextReduce(self) -> 'formula':
        raise Exception('Not implemented')

    def gSubformulae(self) -> 'list[formula]':
        raise Exception('Not implemented')

    def reduce(self) -> 'formula':
        return self

    def afWord(self, word: Word) -> 'formula':
        if (len(word) == 0):
            return self
        firstLetter = word[0]
        restofWord = word[1:]
        return self.af(firstLetter).afWord(restofWord)

    def toString(self) -> str:
        raise Exception('Not implemented')

    def __str__(self):
     return self.toString()

class formulaTrue(formula):
    def __init__(self):
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        return True

    def evaluateTemp(self, word: Word) -> bool:
        return True

    def af(self, letter: Letter) -> 'formula':
        return formulaTrue()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return self
    
    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return []

    def toString(self) -> str:
        return 'tt'

    def __eq__(self, other):
        return isinstance(other, formulaTrue)

    def __hash__(self):
        return 1

class formulaFalse(formula):
    def __init__(self):
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        return False

    def evaluateTemp(self, word: Word) -> bool:
        return False

    def af(self, letter: Letter) -> 'formula':
        return formulaFalse()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return self

    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return []

    def toString(self) -> str:
        return 'ff'
    
    def __eq__(self, other):
        return isinstance(other, formulaFalse)

    def __hash__(self):
        return 2

class formulaSymbol(formula):
    def __init__(self, symbol: str):
        self.symbol = symbol
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        return self.symbol in letter

    def evaluateTemp(self, word: 'list[Letter]') -> bool:
        return self.evaluateProp(word[0])

    def af(self, letter: Letter) -> 'formula':
        if (self.symbol in letter):
            return formulaTrue()
        else:
            return formulaFalse()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return self

    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return []

    def toString(self) -> str:
        return self.symbol

    def __eq__(self, other):
        if not isinstance(other, formulaSymbol): return False
        return self.symbol == other.symbol

    def __hash__(self):
        return self.symbol.__hash__()
    
class formulaNegationSymbol(formula):
    def __init__(self, symbol: str):
        self.symbol: str = symbol
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        return self.symbol not in letter

    def evaluateTemp(self, word: Word) -> bool:
        return self.evaluateProp(word[0])

    def af(self, letter: Letter) -> 'formula':
        if (self.symbol in letter):
            return formulaFalse()
        else:
            return formulaTrue()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return self

    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return []

    def toString(self) -> str:
        return '¬' + self.symbol

    def __eq__(self, other):
        if not isinstance(other, formulaNegationSymbol): return False
        return self.symbol == other.symbol

    def __hash__(self):
        return self.symbol.__hash__() + 1

class formulaAnd(formula):
    def __init__(self, fs: 'list[formula]'):
        self.fs: 'list[formula]' = fs
        formula.__init__(self)

    @staticmethod
    def of(f1: formula, f2: formula) -> formula:
        return formulaAnd([f1, f2]).reduce()
        
    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        for f in self.fs:
            if (not f.evaluateProp(letter)):
                return False
        return True

    def evaluateTemp(self, word: Word) -> bool:
        for f in self.fs:
            if (not f.evaluateTemp(word)):
                return False
        return True

    def af(self, letter: Letter) -> 'formula':
        return formulaAnd([f.af(letter) for f in self.fs]).reduce()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return formulaAnd([f.gSubstitute(gList) for f in self.fs]).reduce()

    def nextReduce(self) -> 'formula':
        return formulaAnd([f.nextReduce() for f in self.fs]).reduce()

    def gSubformulae(self) -> 'list[formula]':
        ret: 'list[formula]' = []
        for f in self.fs:
            ret += f.gSubformulae()
        return ret

    def reduce(self) -> 'formula':
        for f in self.fs:
            if (isinstance(f, formulaFalse)): return formulaFalse()

        combined = self.combineSub()

        withoutTruths = [f for f in combined if not isinstance(f, formulaTrue)]
        if (len(withoutTruths) == 0): return formulaTrue()
        if (len(withoutTruths) == 1): return withoutTruths[0]
        return formulaAnd(withoutTruths)
    
    def combineSub(self) -> 'list[formula]':
        combined: 'list[formula]' = []
        for f in self.fs:
            # Combining all ands into one and
            if (isinstance(f, formulaAnd)):
                fs = f.combineSub()
                for f2 in fs:
                    if (f2 not in combined):
                        combined.append(f2)
            # Don't include or if one of it's subofrmulas is already in the and
            elif (isinstance(f, formulaOr)):
                anyMatch = False
                for f2 in f.fs:
                    if (f2 in self.fs):
                        anyMatch = True
                if not anyMatch and f not in combined:
                    combined.append(f)
            else:
                if (f not in combined):
                    combined.append(f)
        return combined

    def toString(self) -> str:
        return '(' + '/\\'.join([f.toString() for f in self.fs]) + ')'

    def __eq__(self, other):
        if not isinstance(other, formulaAnd): return False
        if (len(self.fs) != len(other.fs)): return False
        return self.fs == other.fs

    def __hash__(self):
        return len(self.fs)

class formulaOr(formula):
    def __init__(self, fs: 'list[formula]'):
        self.fs: 'list[formula]' = fs
        formula.__init__(self)

    @staticmethod
    def of(f1: formula, f2: formula) -> 'formula':
        return formulaOr([f1, f2]).reduce()

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        for f in self.fs:
            if (f.evaluateProp(letter)):
                return True
        return False

    def evaluateTemp(self, word: Word) -> bool:
        for f in self.fs:
            if (f.evaluateTemp(word)):
                return True
        return False

    def af(self, letter: Letter) -> 'formula':
        return formulaOr([f.af(letter) for f in self.fs]).reduce()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return formulaOr([f.gSubstitute(gList) for f in self.fs]).reduce()

    def nextReduce(self) -> 'formula':
        return formulaOr([f.nextReduce() for f in self.fs]).reduce()

    def gSubformulae(self) -> 'list[formula]':
        ret: 'list[formula]' = []
        for f in self.fs:
            ret += f.gSubformulae()
        return ret

    def reduce(self) -> 'formula':
        for f in self.fs:
            if (isinstance(f, formulaTrue)): return formulaTrue()

        combined = self.combineSub()

        withoutFalses = [f for f in combined if not isinstance(f, formulaFalse)]
        if (len(withoutFalses) == 0): return formulaFalse()
        if (len(withoutFalses) == 1): return withoutFalses[0]
        return formulaOr(withoutFalses)

    def combineSub(self) -> 'list[formula]':
        combined: 'list[formula]' = []
        for f in self.fs:
            # Combine all ors into one
            if (isinstance(f, formulaOr)):
                fs = f.combineSub()
                for f2 in fs:
                    if (f2 not in combined):
                        combined.append(f2)
            # if a sub-formula is an and it doesn't make sense to have
            # if one of its subs matches is already included
            elif (isinstance(f, formulaAnd)):
                anyMatch = False
                for f2 in f.fs:
                    if (f2 in self.fs):
                        anyMatch = True
                if not anyMatch and f not in combined:
                    combined.append(f)
            else:
                if (f not in combined):
                    combined.append(f)
        return combined

    def toString(self) -> str:
        return '(' + '\\/'.join([f.toString() for f in self.fs]) + ')'

    def __eq__(self, other):
        if not isinstance(other, formulaOr): return False
        if (len(self.fs) != len(other.fs)): return False
        for i in range(len(self.fs)):
            if self.fs[i] != other.fs[i]: return False
        return True

    def __hash__(self):
        return len(self.fs)

class formulaNext(formula):
    def __init__(self, f1: formula):
        self.f1: formula = f1
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        raise Exception('next only available for temporal formulae')

    def evaluateTemp(self, word: Word) -> bool:
        return self.f1.evaluateTemp(word[1:])

    def af(self, letter: Letter) -> 'formula':
        return self.f1

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return formulaNext(self.f1.gSubstitute(gList))

    def nextReduce(self) -> 'formula':
        return self.f1

    def gSubformulae(self) -> 'list[formula]':
        return self.f1.gSubformulae()

    def toString(self) -> str:
        return 'X' + self.f1.toString()

    def __eq__(self, other):
        if not isinstance(other, formulaNext): return False
        return self.f1 == other.f1

    def __hash__(self):
        return self.f1.__hash__() + 1

class formulaFuture(formula):
    def __init__(self, f1: formula):
        self.f1: formula = f1
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        raise Exception('future only available for temporal formulae')

    def evaluateTemp(self, word: Word) -> bool:
        for k in range(len(word)):
            wk = word[k:]
            if (self.f1.evaluateTemp(wk)): return True
        return False

    def af(self, letter: Letter) -> 'formula':
        return formulaOr.of(self.f1.af(letter), self).reduce()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return formulaFuture(self.f1.gSubstitute(gList))

    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return self.f1.gSubformulae()

    def toString(self) -> str:
        return 'F' + self.f1.toString() 

    def __eq__(self, other):
        if not isinstance(other, formulaFuture): return False
        return self.f1 == other.f1

    def __hash__(self):
        return self.f1.__hash__() + 2

class formulaGlobal(formula):
    def __init__(self, f1: formula):
        self.f1: formula = f1
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        raise Exception('global only available for temporal formulae')

    def evaluateTemp(self, word: Word) -> bool:
        for k in range(len(word)):
            wk = word[k:]
            if (not self.f1.evaluateTemp(wk)): return False
        return True

    def af(self, letter: Letter) -> 'formula':
        return formulaAnd.of(self.f1.af(letter), self).reduce()

    def nextReduce(self) -> 'formula':
        return self

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        if (self.f1 in gList):
            return formulaTrue()
        else:
            return formulaFalse()

    def gSubformulae(self) -> 'list[formula]':
        return [self.f1]

    def toString(self) -> str:
        return 'G' + self.f1.toString() 
    
    def __eq__(self, other):
        if not isinstance(other, formulaGlobal): return False
        return self.f1 == other.f1

    def __hash__(self):
        return self.f1.__hash__() + 3

class formulaUntil(formula):
    def __init__(self, f1: formula, f2: formula):
        self.f1: formula = f1
        self.f2: formula = f2
        formula.__init__(self)

    # letter should be the letter that are true
    def evaluateProp(self, letter: Letter) -> bool:
        raise Exception('until only available for temporal formulae')

    def evaluateTemp(self, word: Word) -> bool:
        # For until at least the first letter must satisfy the first formula
        if (not self.f1.evaluateTemp(word)): return False
        for k in range(len(word)):
            wk = word[k:]
            if (self.f2.evaluateTemp(wk)): return True
            if (not self.f1.evaluateTemp(wk)): return False
        return False

    def af(self, letter: Letter) -> 'formula':
        return formulaOr.of(self.f2.af(letter), formulaAnd.of(self.f1.af(letter), self).reduce()).reduce()

    def gSubstitute(self, gList: 'list[formula]') -> 'formula':
        return formulaUntil(self.f1.gSubstitute(gList), self.f2.gSubstitute(gList))

    def nextReduce(self) -> 'formula':
        return self

    def gSubformulae(self) -> 'list[formula]':
        return self.f1.gSubformulae() + self.f2.gSubformulae()

    def toString(self) -> str:
        return "(" + self.f1.toString() + 'U' + self.f2.toString() + ")"

    def __eq__(self, other):
        if not isinstance(other, formulaUntil): return False
        return self.f1 == other.f1 and self.f2 == other.f2

    def __hash__(self):
        return self.f1.__hash__() + self.f2.__hash__()
