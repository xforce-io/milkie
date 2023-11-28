class Context:
    def __init__(self) -> None:
        self.curQuery = None
        self.instructions = []

    def getCurQuery(self):
        return self.curQuery

    def setCurQuery(self, query):
        self.curQuery = query

    def getCurInstruction(self):
        return None if len(self.instructions) == 0 else self.instructions[-1]