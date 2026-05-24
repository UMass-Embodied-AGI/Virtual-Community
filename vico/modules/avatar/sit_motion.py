import numpy as np
from .replay_motion_module import ReplayMotionModule
from .utils import AvatarState, ActionStatus
import genesis as gs

class SitMotion(ReplayMotionModule):
    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name)

    def start(self, obj=None, position=None):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.position = position
        if self.position is None and obj is None:
            raise NotImplementedError()
        if self.position is None:
            self.position = obj.get_pos()
        self.at_stage = 0
        self.at_frame = 0
    
    def step(self, skip_avatar_animation=False):
        if self.at_frame == 0:
            orientation = (self.position - self.robot.global_trans) * np.array([-1.0, -1.0, 0.0])
            orientation = orientation / np.linalg.norm(orientation)
            self.robot.global_rot = np.array([[orientation[0], -orientation[1], 0], [orientation[1], orientation[0], 0], [0, 0, 1]])
        
        self.at_frame += 1
        if skip_avatar_animation:
            self.at_frame = len(self.data) - 1
        if self.at_frame == len(self.data) - 1:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.base_state = AvatarState.SITTING
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.global_trans = (self.robot.global_rot @ self.robot.base_rot @ self.data[-1][:3]) * np.array([1.0, 1.0, 0.0]) + self.robot.global_trans
            self.robot.pose = self.robot.sit_pose
            self.robot.node_trans = self.robot.sit_node
            self.robot.global_mat = self.robot.sit_mat
            self.robot.global_mat_inv = self.robot.sit_mat_inv
        else:
            self.robot.pose = self.data[self.at_frame]
            self.robot.node_trans = self.node_data[self.at_frame]
            self.robot.global_mat = self.global_mat
            self.robot.global_mat_inv = self.global_mat_inv