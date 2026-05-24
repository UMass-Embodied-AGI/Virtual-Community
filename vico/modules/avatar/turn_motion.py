import numpy as np
import genesis.utils.geom as geom_utils
from .base_motion_module import BaseMotionModule
from .utils import AvatarState, ActionStatus
import genesis as gs

class TurnMotion(BaseMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.turn_frame_limit = 15
    
    def start(self, angle, turn_frame_limit=15, turn_sec_limit=1500):
        self.turn_frame_limit = turn_frame_limit
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        self.angle = angle
        self.turn_sec_limit = turn_sec_limit
    
    def step(self, skip_avatar_animation=False):
        if skip_avatar_animation:
            if self.angle > self.turn_sec_limit:
                self.angle -= self.turn_sec_limit
                rot_xyz = np.array([0.0, 0.0, self.turn_sec_limit])
            elif self.angle < -self.turn_sec_limit:
                self.angle += self.turn_sec_limit
                rot_xyz = np.array([0.0, 0.0, -self.turn_sec_limit])
            else:
                rot_xyz = np.array([0.0, 0.0, self.angle])
                self.robot.action_state = AvatarState.NO_ACTION
                self.robot.action_status = ActionStatus.SUCCEED
        elif self.angle > self.turn_frame_limit:
            self.angle -= self.turn_frame_limit
            rot_xyz = np.array([0.0, 0.0, self.turn_frame_limit])
        elif self.angle < -self.turn_frame_limit:
            self.angle += self.turn_frame_limit
            rot_xyz = np.array([0.0, 0.0, -self.turn_frame_limit])
        else:
            rot_xyz = np.array([0.0, 0.0, self.angle])
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
        rot_mat = geom_utils.quat_to_R(geom_utils.euler_to_quat(rot_xyz))
        self.robot.global_rot = rot_mat @ self.robot.global_rot