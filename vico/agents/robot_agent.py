from .agent import Agent

class RobotAgent(Agent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        self.default_command = self.get_default_command()
        self.command = self.default_command
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)

    def convert_action_to_command(self, action):
        pass

    def get_default_command(self):
        pass
