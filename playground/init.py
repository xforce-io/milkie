import os
import json, logging

from sacred import Experiment
from sacred.observers.base import RunObserver

logger = logging.getLogger(__name__)

NewEnv = os.environ.copy()
NewEnv['SCARF_NO_ANALYTICS'] = 'true'
NewEnv['DO_NOT_TRACK'] = 'true'

class MemObserver(RunObserver):
    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        self.config = config

    def log_metrics(self, metrics_by_name, info):
        if len(metrics_by_name) == 0:
            return

        report = {
            "config" : self.config,
            "metrics" : metrics_by_name
        }
        logger.info(f"exp result {json.dumps(report, default=str)}")

from sacred.observers import FileStorageObserver

def createExperiment():
    ex = Experiment()
    ex.observers.append(FileStorageObserver("my_runs"))
    ex.observers.append(MemObserver())
    return ex