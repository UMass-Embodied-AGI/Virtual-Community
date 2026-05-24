import genesis as gs
import torch
from scipy.spatial.transform import Rotation as R
from ...tools.utils import gs_quat2euler
from .robot_base_controller import RobotBaseController
import numpy as np
import genesis.utils.geom as geom_utils


class DroneController(RobotBaseController):
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
        super().__init__(env=env, scene=scene, name=name, terrain_height_path=terrain_height_path,
                         device=device, dt=dt, ego_view_options=ego_view_options, position=position,
                         config_path=config_path, third_person_camera_resolution=third_person_camera_resolution,
                         debug=debug)
        self.__base_rpm = 457400.
        self.robot = self.env.add_entity(
            type="robot",
            name=self.name,
            morph=gs.morphs.Drone(file=self.env_cfg['urdf_path'],
                                  pos=self.base_init_pos.cpu().numpy(),
                                  quat=self.base_init_quat.cpu().numpy(),
                                  scale=5.),
        )
        self.motor_dofs = [i for i in range(4)]
        self.obs_history_buf = torch.zeros((9,), device=self.device, dtype=gs.tc_float)

    def __get_drone_pos(self) -> torch.Tensor:
        return self.robot.get_pos()

    def __get_drone_vel(self) -> torch.Tensor:
        return self.robot.get_vel()

    def __get_drone_att(self) -> torch.Tensor:
        quat = self.robot.get_quat()
        return geom_utils.quat_to_xyz(quat.cpu().numpy())

    def update_observation(self):
        curr_pos = self.__get_drone_pos()
        curr_vel = self.__get_drone_vel()
        curr_att = torch.tensor(self.__get_drone_att().copy()).to(self.device)
        self.obs_history_buf = torch.cat(
            [
                curr_pos,
                curr_vel,
                curr_att,
            ],
            axis=-1,
        )

    def _init_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float
        )
        self.actions = [0.] * 4
        self.commands = None
        self.vel = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=gs.tc_float
        )

    def set_zero_velocity(self):
        self.robot.control_dofs_velocity([0.0] * self.robot.n_dofs)
        self.robot.set_dofs_velocity([0.0] * self.robot.n_dofs)
        self.robot.set_propellels_rpm([0.0, 0.0, 0.0, 0.0])

    def get_global_xy(self):
        return self.robot.get_pos()[:2].cpu().numpy()

    def get_global_height(self):
        return self.robot.get_pos().cpu().numpy()[-1]

    def get_global_pose(self):
        return np.concatenate([self.robot.get_pos().cpu().numpy(), self.robot.get_quat().cpu().numpy()], axis=0)

    def reset(self, position=None, rotation=None):
        if position is not None:
            self.robot.set_pos(position, zero_velocity=True)
        if len(rotation.shape) > 1:
            rotation = geom_utils.R_to_quat(rotation)
        if rotation is not None:
            self.robot.set_quat(rotation, zero_velocity=True)
        return None, None

    def initialize(self):
        self._init_buffers()
        self.init = True

    def clamp(self, rpm):
        min_rpm = 0.9 * self.__base_rpm
        max_rpm = 1.5 * self.__base_rpm
        return max(min_rpm, min(int(rpm), max_rpm))

    def perform_control(self, action):
        if not self.init:
            self.initialize()
        if action is not None:
            if type(action) == dict and action['type'] == 'control':
                self.actions = action['control']

    def step(self, actions=None, perform_physics_step=False, robot_step=0):
        if not self.init:
            self.initialize()
        actions = self.actions
        if len(actions) < 4:
            import pdb; pdb.set_trace()
        [M1, M2, M3, M4] = actions
        M1 = self.clamp(M1)
        M2 = self.clamp(M2)
        M3 = self.clamp(M3)
        M4 = self.clamp(M4)
        self.robot.set_propellels_rpm([M1, M2, M3, M4])
        self.update_observation()

        return self.obs_history_buf, None, None, None, None

    def get_third_person_camera_rgb(
            self,
            indoor=False
    ):
        global_xy = np.array(self.get_global_xy())
        if indoor:
            pose_x = global_xy[0] - 0.2
            pose_y = global_xy[1] - 0.2
            self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height() + 1.0]),
                                              lookat=np.array(
                                                  [global_xy[0], global_xy[1], self.get_global_height()]))

        else:
            pose_x = global_xy[0] - 0.5
            pose_y = global_xy[1] - 0.5
            self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height() + 0.3]),
                                              lookat=np.array(
                                                  [global_xy[0], global_xy[1], self.get_global_height()]))
        rgb, _, _, _ = self.third_person_camera.render(depth=False)
        return rgb

    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        head_pos = self.robot.get_pos()
        euler_xyz_rad = gs_quat2euler(self.robot.get_quat()).cpu().numpy()
        rot = R.from_euler('xyz', euler_xyz_rad, degrees=False)
        facing_direction = rot.apply((1., 0., 0.))
        facing_direction = facing_direction / np.linalg.norm(facing_direction)
        head_pos = head_pos.cpu().numpy()
        head_pos[2] += 0.5
        head_pos += facing_direction * 0.5  # Move the camera forward by a certain distance to prevent it from seeing the avatar.

        self.ego_view.set_pose(pos=head_pos, lookat=facing_direction + head_pos)
        rgb, depth, seg, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg, self.ego_view.transform
