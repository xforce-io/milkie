from milkie.config.config import ExperienceConfig


class Experience(object):
    def __init__(self, config :ExperienceConfig):
        self.config = config

    def add(self, experience :str):
        pass