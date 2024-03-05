import numpy as np
from omegaconf import DictConfig

from src.optimization.llm.gpt import ChatMessage
from src.optimization.optimizers.base import BaseOptimizer

# TODO
TEST_ASSUMPTION_PIPELINE = ""


class AssumptionOptimizer(BaseOptimizer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def get_completion(self):
        message = ChatMessage("\n".join(self.llm_template.messages))
        np.random.seed()
        if self.cfg.experiment.test_mode:
            # if self.cfg.experiment.interactive_mode:
            # pyperclip.copy(message.content)
            completion = input("Enter completion: ")
        else:
            completion = self.chatbot.get_completion(chat_messages=message)
        return completion
