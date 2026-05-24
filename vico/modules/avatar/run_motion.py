import pdb

import numpy as np
from .replay_motion_module import ReplayMotionModule
from .utils import AvatarState, ActionStatus
import genesis as gs

class RunMotion(ReplayMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_from = np.zeros(3)
        self.run_distance = 0.0

    def start(self, distance, speed=1.0):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.at_frame = 0
        self.run_from = self.robot.global_trans.copy()
        self.run_distance = distance
        self.speed = speed
    
    def step(self, skip_avatar_animation=False):
        self.at_frame += 1
        if skip_avatar_animation:
            if self.run_distance <= self.speed:
                self.robot.action_state = AvatarState.NO_ACTION
                self.robot.action_status = ActionStatus.SUCCEED
                self.robot.global_trans[:2] = (self.robot.global_rot @ (np.array([1.0, 0.0, 0.0]) * self.run_distance))[:2]+ self.robot.global_trans[:2]
                self.robot.pose = self.robot.stop_pose
                self.robot.node_trans = self.robot.stop_node
                self.robot.global_mat = self.robot.stop_mat
                self.robot.global_mat_inv = self.robot.stop_mat_inv
            else:
                self.robot.global_trans[:2] = (self.robot.global_rot @ (np.array([1.0, 0.0, 0.0]) * self.speed))[:2] + self.robot.global_trans[:2]
                self.run_distance -= self.speed
                self.robot.pose = self.robot.stop_pose
                self.robot.node_trans = self.robot.stop_node
                self.robot.global_mat = self.robot.stop_mat
                self.robot.global_mat_inv = self.robot.stop_mat_inv
            return
        if self.at_frame == len(self.data) - 1:
            period_distance = np.linalg.norm(self.data[-1][:3] - self.data[0][:3])
            self.robot.global_trans[:2] = (self.robot.global_rot @ (np.array([1.0, 0.0, 0.0]) * period_distance))[:2] + self.robot.global_trans[:2]
            self.at_frame = 0
        current_global_trans = self.robot.global_rot @ self.robot.base_rot @ (self.data[self.at_frame][:3] - self.data[0][:3]) + self.robot.global_trans
        run_distance = np.linalg.norm((current_global_trans - self.run_from)[:2])
        if (run_distance >= self.run_distance):
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.global_trans[:2] = (self.robot.global_rot @ (np.array([1.0, 0.0, 0.0]) * self.run_distance))[:2] + self.run_from[:2]
            self.robot.pose = self.robot.stop_pose
            self.robot.node_trans = self.robot.stop_node
            self.robot.global_mat = self.robot.stop_mat
            self.robot.global_mat_inv = self.robot.stop_mat_inv
        else:
            self.robot.pose = self.data[self.at_frame]
            self.robot.node_trans = self.node_data[self.at_frame]
            self.robot.global_mat = self.global_mat
            self.robot.global_mat_inv = self.global_mat_inv