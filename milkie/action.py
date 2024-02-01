
from milkie.prompt.prompt_maker import PromptMaker


class ActionModule:
    def __init__(self):
        self.prompt_maker = PromptMaker()

    def act(self, context):
        instruction = context.getCurInstruction()
        if instruction is None:
            return True

    def followInstruction(self, instruction):
        prompt = self.prompt_maker.makePrompt(instruction)