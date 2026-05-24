from .base_motion_module import BaseMotionModule
from .utils import AvatarState, ActionStatus
import numpy as np
import genesis as gs

class ExitBusMotion(BaseMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.IN_VEHICLE:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        # TODO: this is set temporarily to let the avatar get out in the right side. Replace this when motions are ready.
        off_pos = np.array([0.0, -3.0, 0.0])
        self.off_pos = self.robot._h_attach_to.robot.global_trans + self.robot._h_attach_to.robot.global_rot @ off_pos
    
    def step(self, skip_avatar_animation=False):
        self.robot.global_trans[:2] = self.off_pos[:2]
        if self.robot._h_attach_to is not None:
            self.robot._h_attach_to.occupied = False
        self.robot._h_attach_to = None
        self.robot.pose = self.robot.stop_pose
        self.robot.node_trans = self.robot.stop_node
        self.robot.global_mat = self.robot.stop_mat
        self.robot.global_mat_inv = self.robot.stop_mat_inv
        self.robot.action_state = AvatarState.NO_ACTION
        self.robot.base_state = AvatarState.STANDING
        self.robot.action_status = ActionStatus.SUCCEED