import torch
import genesis as gs
import numpy as np
from .robot_base_controller import RobotBaseController
import genesis.utils.geom as geom_utils
from .robot_utils import relative_transform_matrix, quat_to_direction


class HuskyController(RobotBaseController):
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
        self.num_commands = 0
        self.attached_object = {"object": None,
                                "relative_pos": None,
                                "relative_quat": None}

        self.robot = self.env.add_entity(
            type="robot",
            name=self.name,
            morph=gs.morphs.URDF(
                file=self.env_cfg['urdf_path'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=np.array([1., 0., 0., 0.]),
            ),
        )
        self.actions = {'type': 'control', 'arg1': {'position': None,
                                                    "velocity": ([0.0] * (self.robot.n_dofs - 6),
                                                         list(range(self.robot.n_dofs - 6))),
                                                    "force": None}}
        self.motor_dofs = [i for i in range(10)]

        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=gs.tc_float
        )
        self.obs_history_buf = self.obs_history_buf.to(self.device)

    def _init_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float
        )
        self.commands = None
        self.vel = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=gs.tc_float
        )

    def set_zero_velocity(self):
        self.robot.control_dofs_velocity([0.0] * self.robot.n_dofs)
        self.robot.set_dofs_velocity([0.0] * self.robot.n_dofs)

    def attach(self, obj):
        if self.attached_object["object"] is not None:
            return
        self.attached_object["object"] = obj
        self.attached_object["relative_pos"] = obj.get_pos() - self.robot._links[-3].get_pos()
        self.attached_object["relative_quat"] = relative_transform_matrix(self.robot.get_quat().cpu(),
                                                                          obj.get_quat().cpu())
    def update_attached_object(self):
        if self.attached_object["object"] is None:
            return
        pos = self.attached_object["relative_pos"] + self.robot._links[-3].get_pos()
        self.attached_object["object"].set_pos(pos)
        self.attached_object["object"].set_quat(self.robot._links[-3].get_quat())

    def get_global_xy(self):
        return self.robot.get_pos()[:2].cpu().numpy()

    def get_global_height(self):
        return self.robot.get_pos().cpu().numpy()[-1]

    def get_third_person_camera_rgb(
            self,
            indoor=False
    ):
        global_xy = np.array(self.get_global_xy())
        if indoor:
            pose_x = global_xy[0] + 2.5
            pose_y = global_xy[1] - 0.0
            self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height() + 4.0]),
                                              lookat=np.array(
                                                  [global_xy[0], global_xy[1], self.get_global_height()]))

        else:
            pose_x = global_xy[0] + 2.5
            pose_y = global_xy[1] + 0.0
            self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height() + 4.]),
                                              lookat=np.array(
                                                  [global_xy[0], global_xy[1], self.get_global_height()]))
        rgb, _, _, _ = self.third_person_camera.render(depth=False)
        return rgb

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

    def perform_control(self, action):
        if not self.init:
            self.initialize()
        if action is not None:
            if type(action) == dict and action['type'] == 'control':
                self.actions = action

    def step(self, actions=None, perform_physics_step=False, robot_step=0):
        if not self.init:
            self.initialize()
        actions = self.actions
        if actions['arg1']['position'] is not None:
            position_control, position_idx = actions['arg1']['position']
            position_control = torch.tensor(position_control, device=self.device)
            position_idx = torch.tensor(position_idx, device=self.device) + 6
            self.robot.control_dofs_position(position_control, position_idx)
        if actions['arg1']['velocity'] is not None:
            velocity_control, velocity_idx = actions['arg1']['velocity']
            velocity_control = torch.tensor(velocity_control, device=self.device)
            velocity_idx = torch.tensor(velocity_idx, device=self.device) + 6
            self.robot.control_dofs_velocity(velocity_control, velocity_idx)
        if actions['arg1']['force'] is not None:
            force_control, force_idx = actions['arg1']['force']
            force_control = torch.tensor(force_control, device=self.device)
            force_idx = torch.tensor(force_idx, device=self.device) + 6
            self.robot.control_dofs_force(force_control, force_idx)
        self.update_attached_object()
        return self.obs_history_buf, None, None, None, None

    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        head_pos = self.robot.get_pos().cpu().numpy()
        direction = quat_to_direction(self.robot.get_quat().cpu().numpy())
        self.ego_view.set_pose(pos=head_pos + direction / 2, lookat=head_pos + direction)
        rgb, depth, seg, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg, self.ego_view.transform
