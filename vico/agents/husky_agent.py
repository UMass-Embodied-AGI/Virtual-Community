import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
import genesis as gs
import pickle

from .tour_agent import TourAgent
from .agent_utils import get_robot_policy
from .robot_agent import RobotAgent


class HuskyAgent(RobotAgent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)

    def get_default_command(self):
        return {'type': 'control', 'arg1': {'position': None,
                                            "velocity": ([0.0] * 4, [0, 1, 2, 3]),
                                            "force": None}}

    def convert_action_to_command(self, action):
        if action is None:
            return self.default_command
        elif action['type'] == 'move_forward':
            return {'type': 'control', 'arg1': {'position': None,
                                                "velocity": ([10.0] * 4, [0, 1, 2, 3]),
                                                "force": None}}
        elif action['type'] == 'turn_left':
            vel = 10.
            return {'type': 'control', 'arg1': {'position': None,
                                                "velocity": ([-vel, vel, -vel, vel], [0, 1, 2, 3]),
                                                "force": None}}
        elif action['type'] == 'turn_right':
            vel = 10.
            return {'type': 'control', 'arg1': {'position': None,
                                                "velocity": ([vel, -vel, vel, -vel], [0, 1, 2, 3]),
                                                "force": None}}
        elif action['type'] in ['enter', 'force_enter', 'converse']:
            return action
        else:
            assert False, "Unknown action type: {}".format(action['type'])

    def act(self, obs):
        if 'rgb' in obs and obs['rgb'] is not None:
            action = super().act(obs)
            action = self.convert_action_to_command(action)
            self.command = action
        return self.command


class HuskyTourAgent(TourAgent, HuskyAgent):
    def __init__(self, name, pose, info, sim_path, tour_spatial_memory,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path, tour_spatial_memory=tour_spatial_memory,
                         no_react=no_react, debug=debug, logger=logger, **kwargs)
