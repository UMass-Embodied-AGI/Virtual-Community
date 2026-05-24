import numpy as np
from .replay_motion_module import ReplayMotionModule
from .utils import AvatarState, ActionStatus
import genesis as gs

class StandMotion(ReplayMotionModule):
    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name)

    def start(self):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.SITTING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.at_stage = 0
        self.at_frame = 0
    
    def step(self, skip_avatar_animation=False):
        self.at_frame += 1
        if self.at_frame == len(self.data) - 1 or skip_avatar_animation:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.base_state = AvatarState.STANDING
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.global_trans = (self.robot.global_rot @ self.robot.base_rot @ self.data[-1][:3]) * np.array([1.0, 1.0, 0.0]) + self.robot.global_trans
            self.robot.pose = self.robot.stop_pose
            self.robot.node_trans = self.robot.stop_node
            self.robot.global_mat = self.robot.stop_mat
            self.robot.global_mat_inv = self.robot.stop_mat_inv
        else:
            self.robot.pose = self.data[self.at_frame]
            self.robot.node_trans = self.node_data[self.at_frame]
            self.robot.global_mat = self.global_mat
            self.robot.global_mat_inv = self.global_mat_inv