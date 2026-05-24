import numpy as np
import genesis.utils.geom as geom_utils
from .base_motion_module import BaseMotionModule
from .utils import AvatarState, ActionStatus
import genesis as gs

class WakeMotion(BaseMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def start(self):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
    
    def step(self, skip_avatar_animation=False):
        self.robot.base_state = AvatarState.STANDING
        self.robot.action_state = AvatarState.NO_ACTION
        self.robot.action_status = ActionStatus.SUCCEED