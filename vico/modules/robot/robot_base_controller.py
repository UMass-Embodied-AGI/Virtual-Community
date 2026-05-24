import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from ...tools.utils import get_assets_dir
from ...tools.utils import load_height_field
import pickle


class RobotBaseController(ABC):
    def __init__(self,
                 env,
                 scene,
                 name,
                 terrain_height_path,
                 device='cpu',
                 dt=1e-2,
                 ego_view_options=None,
                 position=(0.0, 0.0, 0.0),
                 config_path="",
                 third_person_camera_resolution=None,
                 debug=False
                 ):
        self.debug = debug
        self.env = env
        self.init = False
        self.scene = scene
        self.name = name
        self.device = device
        self.dt = dt
        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = pickle.load(open(config_path, 'rb'))
        self.env_cfg['base_init_pos'] = list(position)
        self.num_envs = 1
        self.num_build_envs = 0
        self.num_single_obs = self.obs_cfg['num_obs']
        self.num_obs = self.num_single_obs * self.obs_cfg['num_history_obs']
        self.base_init_pos = torch.tensor(
            self.env_cfg['base_init_pos'], device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg['base_init_quat'], device=self.device
        )
        self.terrain_height_field = None
        if terrain_height_path:
            self.terrain_height_field = load_height_field(os.path.join(get_assets_dir(), terrain_height_path))
        if ego_view_options is not None:
            self.ego_view = self.scene.add_camera(
                res=ego_view_options["res"],
                pos=(0.0, 0.0, 0.0),
                lookat=(1.0, 0.0, 0.0),
                fov=ego_view_options["fov"],
                GUI=ego_view_options["GUI"],
                far=16000.0,
            )
        else:
            self.ego_view = None
        if third_person_camera_resolution:
            self.third_person_camera = self.scene.add_camera(
                res=(third_person_camera_resolution, third_person_camera_resolution),
                pos=(0.0, 0.0, 0.0),
                lookat=(0.0, 0.0, 0.0),
                fov=90,
                GUI=False,
                far=16000.0,
            )
        self.obs_history_buf = None
        self.robot = None

    def get_pos(self):
        return self.robot.get_pos()

    def get_qpos(self):
        return self.robot.get_qpos()

    def get_quat(self):
        return self.robot.get_quat()

    def perform_control(self, control):
        if control is not None:
            self.actions = control['arg1']

    @abstractmethod
    def _init_buffers(self):
        pass

    def get_observations(self):
        return self.obs_history_buf.detach().cpu().numpy(), None

    @abstractmethod
    def get_global_xy(self):
        pass

    @abstractmethod
    def get_global_height(self):
        pass

    @abstractmethod
    def get_third_person_camera_rgb(self, indoor=False ):
        pass

    def spare(self):
        return True

    def action_status(self):
        return self.robot.action_status

    @abstractmethod
    def get_global_pose(self):
        pass

    @abstractmethod
    def reset(self, position=None, rotation=None):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def step(self, actions=None, perform_physics_step=False, robot_step=0):
        pass

    @abstractmethod
    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        pass
