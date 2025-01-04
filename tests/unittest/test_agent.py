import os
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from milkie.agent.agent import Agent
from milkie.context import Context

class TestAgent(unittest.TestCase):
    def setUp(self):
        config_path = os.path.join(os.path.dirname(__file__), '../../config/global_test.yaml')
        self.context = Context.create(config_path)
        self.toolkit = Mock()

    def testCompile(self):
        code = """
        1. First step
        DEF myFunc(param1, param2):
            2. Do something with {param1}
            3. Do something with {param2}
        END
        4. Second step
        """
        agent = Agent(
            name="test_agent",
            desc="test description",
            code=code,
            context=self.context,
            toolkit=self.toolkit
        )
        agent.compile()