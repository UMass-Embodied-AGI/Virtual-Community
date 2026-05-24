from .base_motion_module import BaseMotionModule
from .utils import AvatarState, ActionStatus
import numpy as np
import genesis as gs

class EnterBusMotion(BaseMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self, bus):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.base_state = AvatarState.IN_VEHICLE
        self.robot.action_status = ActionStatus.ONGOING
        self.bus = bus
        self.bus_pos = bus.robot.global_trans
    
    def step(self, skip_avatar_animation=False):
        self.robot.global_trans[:2] = self.bus_pos[:2]
        self.robot._h_attach_to = self.bus
        self.robot._h_attach_to.occupied = True
        self.robot.action_state = AvatarState.NO_ACTION
        self.robot.action_status = ActionStatus.SUCCEED