from typing import List

class language():
    def __init__(self):
        self.dict = {'': ''}

    def addLabel(self, labels: List[str]):
        label = ', '.join(labels)
        if label == '' or label in self.dict.keys():
            return
        values = self.dict.values()
        if len(label) > 0 and label[0] not in values:
            c = label[0]
        else:
            c = 'a'
            while (c in values):
                i = ord(c[0])
                i += 1
                c = chr(i)
        self.dict[label] = c
    
    def fromLabel(self, labels: List[str]):
        label = ', '.join(labels)
        if label != '' and label not in self.dict.keys():
            self.addLabel(labels)
        return self.dict[label]