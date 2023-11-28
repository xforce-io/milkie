from context import Context


class RetrievalModule:
    def __init__(self, long_term_memory):
        self.long_term_memory = long_term_memory

    def retrieve(self, context :Context):
        return self.long_term_memory.get(context, None)
