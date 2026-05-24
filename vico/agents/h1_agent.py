import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
import genesis as gs
import pickle

from .tour_agent import TourAgent
from .agent_utils import get_robot_policy
from .robot_agent import RobotAgent
from ..tools.utils import get_assets_dir


class H1Agent(RobotAgent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open(os.path.join(get_assets_dir(), "robot_cfgs/h1/cfgs.pkl"), 'rb'))
        self.command_cfg = command_cfg
        self.num_single_obs = obs_cfg['num_obs']
        self.obs_history = torch.zeros([1, self.num_single_obs * 1],
                                       dtype=torch.float, device=gs.backend.name)
        self.policy = get_robot_policy(os.path.join(get_assets_dir(), "ViCo/policies/h1.pt"))
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)

    def get_clock_inputs(self, t):
        t -= int(t / 1000) * 1000
        frequencies = self.command[4]
        phases = self.command[5]

        gait_indices = t * frequencies - int(t * frequencies)
        foot_indices = torch.tensor([phases, 0, ]) + gait_indices

        return torch.sin(2 * np.pi * foot_indices)

    def complete_h1_command(self, commands):
        commands[4] = self.command_cfg['gait_frequency_range'][0]
        commands[7] = (self.command_cfg['body_height_range'][0] + self.command_cfg['body_height_range'][1]) / 2
        commands[8] = self.command_cfg['foot_swing_height_range'][0]
        commands[9] = (self.command_cfg['body_pitch_range'][0] + self.command_cfg['body_pitch_range'][1]) / 2
        commands[10] = (self.command_cfg['body_roll_range'][0] + self.command_cfg['body_roll_range'][1]) / 2
        commands[11] = (self.command_cfg['stance_width_range'][0] + self.command_cfg['stance_width_range'][1]) / 2
        gait_to_id = {
            'jump': 0,
            'walk': 1,
        }
        gait = gait_to_id[self.command_cfg['gaits'][0]]
        if gait == 0:
            pass
        else:
            commands[4 + gait] = 0.5
        return {'type': 'robot', 'command': commands}

    def get_default_command(self):
        commands = np.zeros((12,), dtype=np.float32)
        commands[2] = 0.5
        return self.complete_h1_command(commands)['command']

    def convert_action_to_command(self, action):
        if action['type'] in ['enter', 'force_enter', 'converse']:
            return action
        commands = np.zeros((12,), dtype=np.float32)
        if action['type'] == 'move_forward':
            commands[0] = 0.2
        elif action['type'] == 'turn_left':
            commands[2] = 0.5
        elif action['type'] == 'turn_right':
            commands[2] = -0.5
        return self.complete_h1_command(commands)

    def act(self, obs):
        if 'rgb' in obs and obs['rgb'] is not None:
            action = super().act(obs)
            action = self.convert_action_to_command(action)
            if 'command' in action: # control
                self.command = action['command']
            else: # enter or converse
                return action
        clock_inputs = self.get_clock_inputs(obs['robot_t'])
        obs['robot_obs'] = torch.tensor(obs['robot_obs'], device=gs.backend.name)
        obs['robot_obs'] = torch.concatenate([
            obs['robot_obs'][:, :39],
            clock_inputs[None].to(obs['robot_obs'].device),
        ], dim=-1)
        self.obs_history = torch.concatenate([self.obs_history[:, self.num_single_obs:], obs['robot_obs']], dim=-1)
        control = self.policy(self.obs_history)
        control[0, 5] = 0
        return {'type': 'control', 'control': control}


class H1TourAgent(TourAgent, H1Agent):
    def __init__(self, name, pose, info, sim_path, tour_spatial_memory,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path, tour_spatial_memory=tour_spatial_memory,
                         no_react=no_react, debug=debug, logger=logger, **kwargs)
