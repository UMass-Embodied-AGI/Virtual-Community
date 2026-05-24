import torch
import genesis as gs
import numpy as np
from .robot_base_controller import RobotBaseController
import genesis.utils.geom as geom_utils
from scipy.spatial.transform import Rotation as R
from ...tools.utils import gs_quat2euler
import math
import pickle
from .robot_utils import relative_transform_matrix


class GoogleRobotController(RobotBaseController):
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
        self.looking_down = False
        self.env_cfg, self.obs_cfg, self.reward_cfg, self.command_cfg = pickle.load(open(config_path, 'rb'))
        self.attached_object = {"object": None,
                                "relative_pos": None,
                                "relative_quat": None}

        self.robot = self.env.add_entity(
            type="robot",
            name=self.name,
            morph=gs.morphs.MJCF(
                file=self.env_cfg['urdf_path'],
                pos=(0., 0., 0.),
                quat=(1., 0., 0., 0.),
            ),
        )
        self.actions = {'type': 'control', 'arg1': {'base_control': [0.0, 0.0],
                                                    'position': None,
                                                    "velocity": ([0.0] * (self.robot.n_dofs - 4),
                                                                 list(range(self.robot.n_dofs - 4))),
                                                    "force": None}}
        self.motor_dofs = [i for i in range(self.robot.n_dofs)]
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=gs.tc_float
        ).to(self.device)

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

    def look_down(self):
        self.looking_down = True

    def look_up(self):
        self.looking_down = False

    def get_global_xy(self):
        return self.robot.get_pos()[:2].cpu().numpy()

    def get_global_height(self):
        return self.robot.get_pos().cpu().numpy()[-1]

    def get_third_person_camera_rgb(
            self,
            indoor=False
    ):
        global_xy = np.array(self.get_global_xy())
        pose_x = global_xy[0]
        pose_y = global_xy[1] + 2
        self.third_person_camera.set_pose(pos=np.array([pose_x, pose_y, self.get_global_height() + 1]),
                                          lookat=np.array(
                                              [global_xy[0], global_xy[1] - 2, self.get_global_height() - 1.]))
        rgb, _, _, _ = self.third_person_camera.render(depth=False)
        return rgb

    def get_global_pose(self):
        self.robot.get_quat().cpu().numpy()
        return np.concatenate([self.robot.get_pos().cpu().numpy(),
                               self.get_quat().cpu().numpy()], axis=0)

    def reset(self, position=None, rotation=None):
        if position is not None:
            self.robot.set_pos(torch.tensor(position, device=gs.device, dtype=torch.float), unsafe=True)
        return None, None

    def initialize(self):
        self.robot._links[0].set_friction(0.02)
        self.robot._links[1].set_friction(0.02)
        self.robot._links[3].set_friction(0.02)
        self.robot._links[-2].set_friction(5.0)
        self.robot._links[-1].set_friction(5.0)
        kv = np.array([100, 100, 10, 350, 400, 400, 400, 200, 200, 100, 100, 20, 20])
        self.robot.set_dofs_force_range(torch.tensor([-10000., -10000., -10000.]),
                                        torch.tensor([10000., 10000., 10000.]),
                                        [0, 1, 3])
        self.robot.set_dofs_kp(kv * 100.)
        self.robot.set_dofs_kv(kv * 10.)
        self._init_buffers()
        self.init = True

    def control_base(self, torques):
        euler = self.robot.get_dofs_position([3])[0]
        forward_x = torch.cos(euler)
        forward_y = torch.sin(euler)
        force_forward = torques[0]
        control_force = torch.zeros(4, device=self.device)
        control_force[0] = force_forward * forward_x
        control_force[1] = force_forward * forward_y
        control_force[3] = torques[1]
        self.robot.control_dofs_velocity(control_force, self.motor_dofs[:4])

    def attach(self, obj):
        self.attached_object["object"] = obj
        self.attached_object["relative_pos"] = obj.get_pos() - self.robot._links[-3].get_pos()
        self.attached_object["relative_quat"] = relative_transform_matrix(self.robot._links[-3].get_quat().cpu(),
                                                                          obj.get_quat().cpu())

    def update_attached_object(self):
        if self.attached_object["object"] is None:
            return
        pos = self.attached_object["relative_pos"] + self.robot._links[-3].get_pos()
        self.attached_object["object"].set_pos(pos)
        self.attached_object["object"].set_quat(self.robot._links[-3].get_quat())

    def detach(self):
        self.attached_object["object"] = None
        self.attached_object["relative_pos"] = None
        self.attached_object["relative_quat"] = None

    def perform_control(self, action):
        if not self.init:
            self.initialize()
        if action is not None:
            if type(action) == dict and action['type'] == 'control':
                self.actions = action

    def step(self, actions=None, perform_physics_step=False, robot_step=0):
        if not self.init:
            self.initialize()
        # clip_actions = self.env_cfg['clip_actions']
        actions = self.actions
        if 'set' in actions['arg1'] and actions['arg1']['set'] is not None:
            self.robot.set_qpos(actions['arg1']['set'])
        if actions['arg1']['position'] is not None:
            position_control, position_idx = actions['arg1']['position']
            position_control = torch.tensor(position_control, device=self.device)
            position_idx = torch.tensor(position_idx, device=self.device) + 4
            self.robot.control_dofs_position(position_control, position_idx)
        if actions['arg1']['velocity'] is not None:
            velocity_control, velocity_idx = actions['arg1']['velocity']
            velocity_control = torch.tensor(velocity_control, device=self.device)
            velocity_idx = torch.tensor(velocity_idx, device=self.device) + 4
            self.robot.control_dofs_velocity(velocity_control, velocity_idx)
        if actions['arg1']['force'] is not None:
            force_control, force_idx = actions['arg1']['force']
            force_control = torch.tensor(force_control, device=self.device)
            force_idx = torch.tensor(force_idx, device=self.device) + 4
            self.robot.control_dofs_force(force_control, force_idx)
        self.control_base(actions['arg1']['base_control'])
        self.update_attached_object()
        return self.obs_history_buf, None, None, None, None

    def get_quat(self):
        yaw = self.robot.get_qpos()[3]
        w = math.cos(yaw / 2)
        x = 0.0
        y = 0.0
        z = math.sin(yaw / 2)
        return torch.tensor([w, x, y, z]).to(self.device)

    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        head_pos = self.robot.get_pos()
        euler_xyz_rad = gs_quat2euler(self.get_quat()).cpu().numpy()
        rot = R.from_euler('xyz', euler_xyz_rad, degrees=False)
        facing_direction = rot.apply((1., 0., 0.))
        facing_direction = facing_direction / np.linalg.norm(facing_direction)
        head_pos = head_pos.cpu().numpy()
        head_pos[2] += 1.5
        head_pos += facing_direction * 0.5  # Move the camera forward by a certain distance to prevent it from seeing the avatar.

        if self.looking_down:
            facing_direction[2] -= 1.5
        self.ego_view.set_pose(pos=head_pos, lookat=facing_direction + head_pos)
        rgb, depth, seg, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg, self.ego_view.transform
