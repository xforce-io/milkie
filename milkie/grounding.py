import json

class GroundingModule:
    def __init__(self):
        self.action_data = {}

    def update(self, json_data):
        action_results = json.loads(json_data)
        for action_result in action_results:
            action_type = action_result.get('action_type')
            data = action_result.get('data')
            self.merge_action_data(action_type, data)

    def merge_action_data(self, action_type, data):
        if action_type not in self.action_data:
            self.action_data[action_type] = []
        self.action_data[action_type].append(data)

    def get_data(self):
        return self.action_data
