import numpy as np
import os
import torch
import pickle
from scipy.spatial.transform import Rotation
import genesis as gs

from .tour_agent import TourAgent
from ..modules.robot.robot_utils import gs_quat2euler
from .robot_agent import RobotAgent
from ..tools.utils import get_assets_dir


class GoogleRobotAgent(RobotAgent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open(os.path.join(get_assets_dir(), "robot_cfgs/google_robot/cfgs.pkl"), 'rb'))
        self.num_single_obs = obs_cfg['num_obs']
        self.obs_history = torch.zeros([1, self.num_single_obs * obs_cfg['num_history_obs']],
                                       dtype=torch.float, device=gs.backend.name)
        self.arm_dofs = env_cfg['arm_dofs']
        self.base_dofs = env_cfg['base_dofs']
        self.holding_object = False
        self.goal_condition = None
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)

    def calculate_goal_condition(self, obs, action):
        if action['type'] == 'move_forward':
            facing = Rotation.from_euler('xyz', obs['pose'][3:], degrees=False).apply(np.array([1., 0., 0.]))
            facing = facing / np.linalg.norm(facing)
            self.goal_condition = {'type': 'move_forward',
                                   'facing': facing,
                                   'original': obs['pose'][:3],
                                   'arg1': action['arg1']}
        elif action['type'] in ['turn_left', 'turn_right']:
            self.goal_condition = {'type': action['type'],
                                   'original': obs['pose'][3:],
                                   'arg1': action['arg1']}
        else:
            self.goal_condition = None

    def check_if_meet_goal_condition(self, obs, degree_error_threshold=20):
        if self.goal_condition is None:
            return False
        if self.goal_condition['type'] == 'move_forward':
            proj_len = np.dot(self.goal_condition['facing'], obs['pose'][:3] - self.goal_condition['original'][:3])
            gs.logger.info(f"DEBUG {proj_len}, {self.goal_condition['arg1']}")
            return proj_len > self.goal_condition['arg1']
        elif self.goal_condition['type'] in ['turn_left', 'turn_right']:
            degree_diff = (torch.rad2deg(gs_quat2euler(torch.tensor(self.goal_condition['original'])))[2] -
                           torch.rad2deg(gs_quat2euler(torch.tensor(obs['pose'][3:])))[2])
            if self.goal_condition['type'] == 'turn_left':
                degree_diff = -degree_diff
            gs.logger.info(f"DEBUG {float(self.goal_condition['arg1'])}, {float(degree_diff % 360)}, {float(degree_diff)}")
            return self.goal_condition['arg1'] < degree_diff % 360 < 360 - degree_error_threshold
        return False

    def get_default_command(self):
        arm_dofs_idx = list(range(self.arm_dofs))
        gripper_force= 0.0 if not self.holding_object else 0.5
        return {'type': 'control', 'arg1': {'base_control': [0.0, 0.0],
                                               'position': None,
                                               "velocity": ([0.0] * (self.arm_dofs - 2), arm_dofs_idx[:-2]),
                                               "force": ([gripper_force] * 2, arm_dofs_idx[-2:])}}

    def convert_action_to_command(self, action):
        arm_dofs_idx = list(range(self.arm_dofs))
        gripper_force= 0.0 if not self.holding_object else 0.5
        if action is None:
            return self.get_default_command()
        elif action['type'] == 'move_forward':
            return {'type': 'control', 'arg1': {'base_control': [10.0, 0.0],
                                                   'position': None,
                                                   "velocity": ([0.0] * (self.arm_dofs - 2), arm_dofs_idx[:-2]),
                                                   "force": ([gripper_force] * 2, arm_dofs_idx[-2:])}}
        elif action['type'] == 'turn_left':
            return {'type': 'control', 'arg1': {'base_control': [0.0, 3.0],
                                                   'position': None,
                                                   "velocity": ([0.0] * (self.arm_dofs - 2), arm_dofs_idx[:-2]),
                                                   "force": ([gripper_force] * 2, arm_dofs_idx[-2:])}}
        elif action['type'] == 'turn_right':
            return {'type': 'control', 'arg1': {'base_control': [0.0, -3.0],
                                                   'position': None,
                                                   "velocity": ([0.0] * (self.arm_dofs - 2), arm_dofs_idx[:-2]),
                                                   "force": ([gripper_force] * 2, arm_dofs_idx[-2:])}}
        elif action['type'] in ['enter', 'force_enter', 'converse']:
            return action
        else:
            return self.get_default_command()

    def act(self, obs):
        if 'rgb' in obs and obs['rgb'] is not None:
            action = super().act(obs)
            gs.logger.info(f"DEBUG {action}")
            self.calculate_goal_condition(obs, action)
            self.command = self.convert_action_to_command(action)
        if self.check_if_meet_goal_condition(obs):
            return self.default_command
        return self.command


class GoogleRobotTourAgent(TourAgent, GoogleRobotAgent):
    def __init__(self, name, pose, info, sim_path, tour_spatial_memory,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path, tour_spatial_memory=tour_spatial_memory,
                         no_react=no_react, debug=debug, logger=logger, **kwargs)
