from .avatar_robot import AvatarRobot
import genesis as gs

class BaseMotionModule():
    def __init__(self, motion_name: str, robot: AvatarRobot):
        self.motion_name = motion_name
        self.robot = robot