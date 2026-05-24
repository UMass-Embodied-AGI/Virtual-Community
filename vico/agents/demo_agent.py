import copy

class DemoAgent:
    def __init__(self, action_list):
        self.action_buf = copy.deepcopy(action_list)

    def reset(self, action_list):
        self.action_buf = copy.deepcopy(action_list)

    def act(self, obs):
        if obs['action_status'] == "ONGOING":
            return None
        return self.action_buf.pop(0) if len(self.action_buf) > 0 else None
