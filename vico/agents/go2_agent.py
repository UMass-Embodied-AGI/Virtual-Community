import numpy as np
import os
import torch
import pickle
from scipy.spatial.transform import Rotation
import genesis as gs

from .tour_agent import TourAgent
from .agent_utils import get_robot_policy
from .robot_agent import RobotAgent
from ..tools.utils import get_assets_dir


class Go2Agent(RobotAgent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(open(os.path.join(get_assets_dir(), "robot_cfgs/go2/cfgs.pkl"), 'rb'))
        self.num_single_obs = obs_cfg['num_obs']
        self.obs_history = torch.zeros([1, self.num_single_obs * obs_cfg['num_history_obs']],
                                       dtype=torch.float, device=gs.backend.name)
        self.policy = get_robot_policy(os.path.join(get_assets_dir(), "ViCo/policies/go2.pt"))
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)

    def get_clock_inputs(self, t):
        t -= int(t)
        frequencies = self.command[4]
        phases = self.command[5]
        offsets = self.command[6]
        bounds = self.command[7]

        gait_indices = t * frequencies - int(t * frequencies)
        foot_indices = torch.tensor([phases + offsets + bounds, offsets, bounds, phases]) + gait_indices

        return torch.sin(2 * np.pi * foot_indices)

    def get_default_command(self):
        return np.array([
            0.0, 0.0, 0.0, 0.0,
            2.0, 0.5, 0.0, 0.0, 0.5,  # frequencies, phases, offsets, bounds, duration
            0.3, 0.15, 0.0, 0.0,  # body height, foot swing height, body pitch, body roll
            0.3, 0.4,  # stance width, stance length
        ], dtype=np.float32)

    def convert_action_to_command(self, action):
        if action is None:
            return {'type': 'robot',
                    'command': np.array([
                        0.0, 0.0, 0.0, 0.0,
                        2.0, 0.5, 0.0, 0.0, 0.5,  # frequencies, phases, offsets, bounds, duration
                        0.3, 0.15, 0.0, 0.0,  # body height, foot swing height, body pitch, body roll
                        0.3, 0.4,  # stance width, stance length
                    ], dtype=np.float32)}
        elif action['type'] == 'move_forward':
            return {'type': 'robot',
                    'command': np.array([
                        0.8, 0.0, 0.0, 0.0,
                        2.0, 0.5, 0.0, 0.0, 0.5,
                        0.3, 0.15, 0.0, 0.0,
                        0.3, 0.4,
                    ], dtype=np.float32)}
        elif action['type'] == 'turn_left':
            return {'type': 'robot',
                    'command': np.array([
                        0.0, 0.0, 0.8, 0.0,
                        2.0, 0.5, 0.0, 0.0, 0.5,  # frequencies, phases, offsets, bounds, duration
                        0.3, 0.15, 0.0, 0.0,  # body height, foot swing height, body pitch, body roll
                        0.3, 0.4,  # stance width, stance length
                    ], dtype=np.float32)}
        elif action['type'] == 'turn_right':
            return {'type': 'robot',
                    'command': np.array([
                        0.0, 0.0, -0.8, 0.0,
                        2.0, 0.5, 0.0, 0.0, 0.5,  # frequencies, phases, offsets, bounds, duration
                        0.3, 0.15, 0.0, 0.0,  # body height, foot swing height, body pitch, body roll
                        0.3, 0.4,  # stance width, stance length
                    ], dtype=np.float32),
                    }
        elif action['type'] in ['enter', 'force_enter', 'converse']:
            return action
        else:
            assert False, "Unknown action type: {}".format(action['type'])

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
            obs['robot_obs'][:, -self.num_single_obs:-self.num_single_obs + 45],
            torch.tensor(self.command)[None, 4:].to(obs['robot_obs'].device),
            clock_inputs[None].to(obs['robot_obs'].device)
        ], dim=-1)
        self.obs_history = torch.concatenate([self.obs_history[:, self.num_single_obs:], obs['robot_obs']], dim=-1)
        control = self.policy(self.obs_history)
        return {'type': 'control', 'control': control}


class Go2TourAgent(TourAgent, Go2Agent):
    def __init__(self, name, pose, info, sim_path, tour_spatial_memory,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path, tour_spatial_memory=tour_spatial_memory,
                         no_react=no_react, debug=debug, logger=logger, **kwargs)
