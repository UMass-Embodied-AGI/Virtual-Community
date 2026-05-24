from .go2_controller import Go2Controller
from .h1_controller import H1Controller
from .google_robot_controller import GoogleRobotController
from .drone_controller import DroneController
from .husky_controller import HuskyController

ROBOT_CONTROLLERS = {
    "go2": Go2Controller,
    "h1": H1Controller,
    "google_robot": GoogleRobotController,
    "drone": DroneController,
    "husky": HuskyController
}

ROBOT_POSITION_OFFSETS = {
    "go2": (0, 0, 0.8),
    "h1": (0, 0, 1.4),
    "google_robot": (0, 0, 0.8),
    "drone": (0, 0, 3.0),
    "husky": (0, 0, 0.9)
}

ROBOT_CONFIGS = {
    "go2": "go2/cfgs.pkl",
    "h1": "h1/cfgs.pkl",
    "google_robot": "google_robot/cfgs.pkl",
    "drone": "drone/cfgs.pkl",
    "husky": "husky/cfgs.pkl"
}
