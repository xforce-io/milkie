import unittest
import shutil
import os

class BaseIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.baseDataDir = os.path.abspath(os.path.join('data', 'integration_test'))
        self.prepairScenarioDataDir(self.baseDataDir)
    
    def tearDown(self):
        pass
    
    def getScenarioDataDir(self, scenarioName):
        return os.path.join(self.baseDataDir, scenarioName)

    def prepairScenarioDataDir(self, path):
        baseName = os.path.basename(path)
        if baseName == "." or baseName == ".." or baseName == "":
            return

        try:
            shutil.rmtree(os.path.join("/tmp", baseName), ignore_errors=True)
            shutil.move(path, "/tmp")
        except Exception as e:
            pass
        os.makedirs(path, exist_ok=True)