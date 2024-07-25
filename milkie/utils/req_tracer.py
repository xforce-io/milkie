import json

class ReqTracer:
    def __init__(self) -> None:
        self.log = {}

    def set(self, path: str, val):
        keys = path.split('.')
        for key in keys[:-1]:
            if key not in self.log:
                self.log[key] = {}
            self.log = self.log[key]
        self.log[keys[-1]] = val

    def add(self, path, val):
        keys = path.split('.')
        for key in keys[:-1]:
            if key not in self.log:
                self.log[key] = {}
            self.log = self.log[key]
        if keys[-1] not in self.log:
            self.log[keys[-1]] = []
        self.log[keys[-1]].append(val)

    def dump(self) -> str:
        return json.dumps(self.log, ensure_ascii=False)