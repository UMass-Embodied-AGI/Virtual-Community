import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
import genesis as gs
import pickle

from .tour_agent import TourAgent
from .agent_utils import get_robot_policy
from .robot_agent import RobotAgent


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

class DroneAgent(RobotAgent):
    def __init__(self, name, pose, info, sim_path,
                 no_react=False, debug=False, logger=None, **kwargs):
        self.__base_rpm = 457400.
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path,
                         no_react=no_react, debug=debug, logger=logger)
        pid_params = [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [20.0, 0.0, 20.0],
            [20.0, 0.0, 20.0],
            [25.0, 0.0, 20.0],
            [10.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [2.0, 0.0, 0.2],
        ]
        self.__pid_pos_x = PIDController(kp=pid_params[0][0], ki=pid_params[0][1], kd=pid_params[0][2])
        self.__pid_pos_y = PIDController(kp=pid_params[1][0], ki=pid_params[1][1], kd=pid_params[1][2])
        self.__pid_pos_z = PIDController(kp=pid_params[2][0], ki=pid_params[2][1], kd=pid_params[2][2])

        self.__pid_vel_x = PIDController(kp=pid_params[3][0], ki=pid_params[3][1], kd=pid_params[3][2])
        self.__pid_vel_y = PIDController(kp=pid_params[4][0], ki=pid_params[4][1], kd=pid_params[4][2])
        self.__pid_vel_z = PIDController(kp=pid_params[5][0], ki=pid_params[5][1], kd=pid_params[5][2])

        self.__pid_att_roll = PIDController(kp=pid_params[6][0], ki=pid_params[6][1], kd=pid_params[6][2])
        self.__pid_att_pitch = PIDController(kp=pid_params[7][0], ki=pid_params[7][1], kd=pid_params[7][2])
        self.__pid_att_yaw = PIDController(kp=pid_params[8][0], ki=pid_params[8][1], kd=pid_params[8][2])
        self.__dt = 1e-2
        self.vel = None
        self.euler = np.array([0., 0., 0.])
        self.curr_drone_target = None
        self.robot_obs = None
        self.command = self.default_command['control']

    def get_default_command(self):
        return {'type': 'control', 'control': [self.__base_rpm] * 4}

    def convert_action_to_command(self, action):
        if action['type'] in ['move_forward', 'turn_left', 'turn_right']:
            pass # drone needs 'move_to' actions
        elif action['type'] in ['enter', 'force_enter', 'converse']:
            return action
        elif action['type'] in ['move_to']:
            self.curr_drone_target = action['control']
        else:
            assert False, "Unknown action type: {}".format(action['type'])
        if self.curr_drone_target is None:
            return self.default_command
        obs = self.robot_obs
        assert len(obs) == 9
        curr_pos = obs[:3]
        curr_vel = obs[3:6]
        curr_att = obs[6:9]
        target = self.curr_drone_target

        err_pos_x = target[0] - curr_pos[0]
        err_pos_y = target[1] - curr_pos[1]
        err_pos_z = target[2] - curr_pos[2]

        vel_des_x = self.__pid_pos_x.update(err_pos_x, self.__dt)
        vel_des_y = self.__pid_pos_y.update(err_pos_y, self.__dt)
        vel_des_z = self.__pid_pos_z.update(err_pos_z, self.__dt)

        error_vel_x = vel_des_x - curr_vel[0]
        error_vel_y = vel_des_y - curr_vel[1]
        error_vel_z = vel_des_z - curr_vel[2]

        x_vel_del = self.__pid_vel_x.update(error_vel_x, self.__dt)
        y_vel_del = self.__pid_vel_y.update(error_vel_y, self.__dt)
        thrust_des = self.__pid_vel_z.update(error_vel_z, self.__dt)

        err_roll = 0.0 - curr_att[0]
        err_pitch = 0.0 - curr_att[1]
        err_yaw = 0.0 - curr_att[2]

        roll_del = self.__pid_att_roll.update(err_roll, self.__dt)
        pitch_del = self.__pid_att_pitch.update(err_pitch, self.__dt)
        yaw_del = self.__pid_att_yaw.update(err_yaw, self.__dt)

        prop_rpms = self.__mixer(thrust_des, roll_del, pitch_del, yaw_del, x_vel_del, y_vel_del)
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()
        prop_rpms = prop_rpms.cpu()
        prop_rpms - prop_rpms.numpy()
        return {'type': 'control', 'control': prop_rpms}

    def act(self, obs):
        self.robot_obs = obs['robot_obs']
        if 'rgb' in obs and obs['rgb'] is not None:
            action = super().act(obs)
            action = self.convert_action_to_command(action)
            if 'control' in action: # control
                self.command = action['control']
            else: # enter or converse
                return action
        return {'type': 'control', 'control': self.command}


class DroneTourAgent(TourAgent, DroneAgent):
    def __init__(self, name, pose, info, sim_path, tour_spatial_memory,
                 no_react=False, debug=False, logger=None, **kwargs):
        super().__init__(name=name, pose=pose, info=info, sim_path=sim_path, tour_spatial_memory=tour_spatial_memory,
                         no_react=no_react, debug=debug, logger=logger, **kwargs)
