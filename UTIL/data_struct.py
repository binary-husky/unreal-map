

class UniqueList():
    def __init__(self, list_input=None):
        self._list = []
        if list_input is not None:
            self.extend_unique(list_input)
    
    def append_unique(self, item):
        if item in self._list:
            return False
        else:
            self._list.append(item)
    
    def extend_unique(self, list_input):
        for item in list_input:
            self.append_unique(item)
    
    def has(self, item):
        return (item in self._list)
    
    def len(self):
        return len(self._list)

    def get(self):
        return self._list