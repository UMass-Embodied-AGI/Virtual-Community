import torch
import genesis as gs
import numpy as np
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
import genesis.utils.geom as geom_utils
from .robot_base_controller import RobotBaseController
from .robot_utils import (gs_quat_mul, gs_inv_quat, gs_quat2euler, gs_quat_from_angle_axis, gs_transform_by_quat,
                          wrap_to_pi, gs_rand_float, gs_euler2quat, gs_quat_apply_yaw, gs_quat_conjugate)


class Go2Controller(RobotBaseController):
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
        self.eval = True
        self.num_privileged_obs = self.obs_cfg['num_priv_obs']
        self.num_actions = self.env_cfg['num_actions']
        self.num_commands = self.command_cfg['num_commands']
        self.headless = True

        self.dt = 1 / self.env_cfg['control_freq']
        self.max_episode_length_s = self.env_cfg['episode_length_s']
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.obs_scales = self.obs_cfg['obs_scales']
        self.reward_scales = self.reward_cfg['reward_scales']

        self.command_type = self.env_cfg['command_type']
        assert self.command_type in ['heading', 'ang_vel_yaw']

        self.action_latency = self.env_cfg['action_latency']
        assert self.action_latency in [0, 0.02]
        self.num_dof = len(self.env_cfg['dof_names'])

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver
            break

        self.robot = self.env.add_entity(
            type="robot",
            name=self.name,
            morph=gs.morphs.URDF(
                file=self.env_cfg['urdf_path'],
                merge_fixed_links=True,
                links_to_keep=self.env_cfg['links_to_keep'],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.num_obs), dtype=gs.tc_float
        ).to(self.device)

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            np.array([0.0, 0.0, -1.0]), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float
        )
        self.obs_noise = torch.zeros(
            (self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float
        )
        self._prepare_obs_noise()
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_pos = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf_neg = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.time_out_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )

        # commands
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales['lin_vel'],
                self.obs_scales['lin_vel'],
                self.obs_scales['ang_vel'],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.stand_still = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )

        # names to indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg['dof_names']
        ]

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(
            self.env_cfg['termination_contact_link_names']
        )
        self.penalized_contact_link_indices = find_link_indices(
            self.env_cfg['penalized_contact_link_names']
        )
        self.feet_link_indices = find_link_indices(
            self.env_cfg['feet_link_names']
        )
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0
        self.feet_link_indices_world_frame = [i + 1 for i in self.feet_link_indices]

        # actions
        self.actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_last_actions = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_pos = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.last_dof_vel = torch.zeros(
            (self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float
        )
        self.root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.last_root_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_int,
        )

        # extras
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.common_step_counter = 0
        self.extras = {}

        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        # self.terrain_heights = self.terrain_height_field(self.base_init_pos[:2].cpu().numpy())
        # self.terrain_heights = torch.tensor(self.terrain_heights, device=self.device, dtype=gs.tc_float)

        # PD control
        stiffness = self.env_cfg['PD_stiffness']
        damping = self.env_cfg['PD_damping']

        self.p_gains, self.d_gains = [], []
        for dof_name in self.env_cfg['dof_names']:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg['default_joint_angles']
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg['dof_names']],
            device=self.device,
        )

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                    m - 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )
            self.dof_pos_limits[i, 1] = (
                    m + 0.5 * r * self.reward_cfg['soft_dof_pos_limit']
            )

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

        # gait control
        self.gait_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=gs.tc_float,
        )
        self.clock_inputs = torch.zeros(
            self.num_envs, len(self.feet_link_indices), device=self.device, dtype=gs.tc_float,
        )
        self.doubletime_clock_inputs = torch.zeros(
            self.num_envs, len(self.feet_link_indices), device=self.device, dtype=gs.tc_float,
        )
        self.halftime_clock_inputs = torch.zeros(
            self.num_envs, len(self.feet_link_indices), device=self.device, dtype=gs.tc_float,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs, len(self.feet_link_indices), device=self.device, dtype=gs.tc_float,
        )

        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 4, device=self.device, dtype=gs.tc_float,
        )
        self.prev_foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        self.prev_foot_velocities = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )

        self.base_link_index = 1

        self.com = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=gs.tc_float,
        )

    def _update_buffers(self):

        # update buffers
        # [:] is for non-parallelized scene
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(self.base_quat,
                                    gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(-self.base_euler[:, 2],
                                               torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat
        )

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.com[:] = self.rigid_solver.get_links_root_COM([self.base_link_index, ]).squeeze(dim=1)

        self.prev_foot_positions = self.foot_positions.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame)
        self.foot_velocities = (self.foot_positions - self.prev_foot_positions) / self.dt
        self.foot_velocities[
            self.reset_buf.nonzero(as_tuple=False).flatten()] = 0  # set velocities in envs reset last step to 0

        # if self.env_cfg['use_terrain']:
        #     clipped_base_pos = self.base_pos[:, :2].clamp(min=torch.zeros(2, device=self.device),
        #                                                   max=self.terrain_margin)
        #     height_field_ids = (clipped_base_pos / self.terrain_cfg['horizontal_scale'] - 0.5).floor().int()
        #     height_field_ids.clamp(min=0)
        #     # print(self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]])
        #     self.terrain_heights = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg['action_scale']
        torques = (
                self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
                - self.batched_d_gains * self.dof_vel
        )
        # print("front", torques[:, :6].detach().cpu().abs().max())
        # print("rear", torques[:, 6:].detach().cpu().abs().max())
        # print(torques.detach().cpu().abs().max())

        return torques * self.motor_strengths

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
                self.episode_length_buf > self.max_episode_length
        )  # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg['termination_if_pitch_greater_than'],
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg['termination_if_roll_greater_than'],
        )
        # print(self.base_pos[:, 2])
        # print(self.base_euler[:, 1])
        # if self.env_cfg['use_terrain']:
        #     self.reset_buf |= torch.logical_or(
        #         self.base_pos[:, 0] > self.terrain_margin[0],
        #         self.base_pos[:, 1] > self.terrain_margin[1],
        #     )
        #     self.reset_buf |= torch.logical_or(
        #         self.base_pos[:, 0] < 1,
        #         self.base_pos[:, 1] < 1,
        #     )
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg['termination_if_height_lower_than']
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
        if self.reward_cfg['only_positive_rewards']:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        elif self.reward_cfg['rewards.only_positive_rewards_ji22_style']:
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.reward_cfg['sigma_rew_neg'])
        # add termination reward after clipping
        if 'termination' in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales['termination']
            self.rew_buf += rew
            self.episode_sums['termination'] += rew

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()
        self._step_contact_targets()

        resampling_time_s = self.env_cfg['resampling_time_s']
        envs_idx = (
            (self.episode_length_buf % int(resampling_time_s / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)
        if self.command_type == 'heading':
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        # random push
        push_interval_s = self.env_cfg['push_interval_s']
        if push_interval_s > 0 and not (self.debug or self.eval):
            max_push_vel_xy = self.env_cfg['max_push_vel_xy']
            dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

        self.check_termination()
        self.compute_reward()

        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.reset_idx(envs_idx)
        # self.rigid_solver.forward_kinematics() # no need currently
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot.get_vel()

    def compute_observations(self):
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales['ang_vel'],  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],  # 12
                self.dof_vel * self.obs_scales['dof_vel'],  # 12
                self.actions,  # 12
                self.commands[:, 4:],  # 12
                self.clock_inputs,  # 4
            ],
            axis=-1,
        )

        # add noise
        if not self.eval:
            self.obs_buf += gs_rand_float(
                -1.0, 1.0, (self.num_single_obs,), self.device
            ) * self.obs_noise

        # self.obs_buf[:, 21:33] = 0.0

        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf = torch.cat(
            [self.obs_history_buf[:, self.num_single_obs:], self.obs_buf.detach()], dim=1
        )

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.base_lin_vel * self.obs_scales['lin_vel'],  # 3
                    self.base_ang_vel * self.obs_scales['ang_vel'],  # 3
                    self.projected_gravity,  # 3
                    self.commands[:, :3] * self.commands_scale,  # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],  # 12
                    self.dof_vel * self.obs_scales['dof_vel'],  # 12
                    self.actions,  # 12
                    self.last_actions,  # 12
                    self.commands[:, 4:],  # 12
                    self.clock_inputs,  # 4
                ],
                axis=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def get_global_xy(self):
        return self.robot.get_pos()[:2].cpu().numpy()

    def get_global_height(self):
        return self.robot.get_pos().cpu().numpy()[-1]

    def get_third_person_camera_rgb(
            self,
            indoor=False
    ):

        head_pos = self.robot.get_pos()
        facing_direction = self.gs_transform_by_quat(self.robot.get_quat())
        head_pos = head_pos.cpu().numpy()
        head_pos += np.array([0., 0., 1.0])
        cam_pos = head_pos - facing_direction * 3.0 + np.array([0., 0., 0.5])
        up = np.array([0., 0., 1.], dtype=head_pos.dtype)
        T = geom_utils.pos_lookat_up_to_T(cam_pos, head_pos, up)
        self.third_person_camera.set_pose(transform=T)
        rgb, _, _, _ = self.third_person_camera.render(depth=False)
        return rgb

    def _prepare_obs_noise(self):
        self.obs_noise[:3] = self.obs_cfg['obs_noise']['ang_vel']
        self.obs_noise[3:6] = self.obs_cfg['obs_noise']['gravity']
        self.obs_noise[21:33] = self.obs_cfg['obs_noise']['dof_pos']
        self.obs_noise[33:45] = self.obs_cfg['obs_noise']['dof_vel']

    def _resample_commands(self, envs_idx):
        # resample commands

        # lin_vel
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg['lin_vel_x_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg['lin_vel_y_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, :2] *= (
                torch.norm(self.commands[envs_idx, :2], dim=1) > 0.2
        ).unsqueeze(1)

        # ang_vel
        if self.command_type == 'heading':
            self.commands[envs_idx, 3] = gs_rand_float(
                -3.14, 3.14, (len(envs_idx),), self.device
            )
        elif self.command_type == 'ang_vel_yaw':
            self.commands[envs_idx, 2] = gs_rand_float(
                *self.command_cfg['ang_vel_range'], (len(envs_idx),), self.device
            )
            self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > 0.2

        # gait frequency
        self.commands[envs_idx, 4] = gs_rand_float(
            *self.command_cfg['gait_frequency_range'], (len(envs_idx),), self.device
        )

        # gait
        random_env_floats = gs_rand_float(
            0.0, 1.0, (len(envs_idx),), self.device
        )
        probability_per_category = 1. / len(self.command_cfg['gaits'])

        for i in range(len(self.command_cfg['gaits'])):
            category_envs_idx = envs_idx[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                           random_env_floats < probability_per_category * (i + 1))]
            if self.command_cfg['gaits'][i] == 'pronk':
                self.commands[category_envs_idx, 5] = 0.
                self.commands[category_envs_idx, 6] = 0.
                self.commands[category_envs_idx, 7] = 0.
            elif self.command_cfg['gaits'][i] == 'trot':
                self.commands[category_envs_idx, 5] = 0.5
                self.commands[category_envs_idx, 6] = 0.
                self.commands[category_envs_idx, 7] = 0.
            elif self.command_cfg['gaits'][i] == 'bound':
                self.commands[category_envs_idx, 5] = 0.
                self.commands[category_envs_idx, 6] = 0.5
                self.commands[category_envs_idx, 7] = 0.
            elif self.command_cfg['gaits'][i] == 'pace':
                self.commands[category_envs_idx, 5] = 0.
                self.commands[category_envs_idx, 6] = 0.
                self.commands[category_envs_idx, 7] = 0.5
            elif self.command_cfg['gaits'][i] == 'run':
                self.commands[category_envs_idx, 5] = 0.
                self.commands[category_envs_idx, 6] = 0.5
                self.commands[category_envs_idx, 7] = 0.1

        # gait duration
        self.commands[envs_idx, 8] = gs_rand_float(
            *self.command_cfg['gait_duration_range'], (len(envs_idx),), self.device
        )

        # body height
        self.commands[envs_idx, 9] = gs_rand_float(
            *self.command_cfg['body_height_range'], (len(envs_idx),), self.device
        )

        # foot swing height
        self.commands[envs_idx, 10] = gs_rand_float(
            *self.command_cfg['foot_swing_height_range'], (len(envs_idx),), self.device
        )
        # self.commands[envs_idx, 10] *= self.commands[envs_idx, 10] > 0.01
        self.commands[envs_idx, 10] = torch.clip(self.commands[envs_idx, 10], max=self.commands[envs_idx, 9] - 0.15)

        # body pitch & roll
        self.commands[envs_idx, 11] = gs_rand_float(
            *self.command_cfg['body_pitch_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 12] = gs_rand_float(
            *self.command_cfg['body_roll_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 11] *= torch.abs(self.commands[envs_idx, 11]) > 0.1
        self.commands[envs_idx, 12] *= torch.abs(self.commands[envs_idx, 12]) > 0.1

        # stance width & length
        self.commands[envs_idx, 13] = gs_rand_float(
            *self.command_cfg['stance_width_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 14] = gs_rand_float(
            *self.command_cfg['stance_height_range'], (len(envs_idx),), self.device
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = (
                                     self.default_dof_pos
                                 ) + gs_rand_float(-0.3, 0.3, (len(envs_idx), self.num_dof), self.device) * 0
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, :2] += gs_rand_float(
            -1.0, 1.0, (len(envs_idx), 2), self.device
        ) * 0
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(
            -0.1, 0.1, (len(envs_idx), 3), self.device
        ) * 0
        base_euler[:, 2] = gs_rand_float(0.0, 3.14, (len(envs_idx),), self.device) * 0
        self.base_quat[envs_idx] = gs_quat_mul(
            gs_euler2quat(base_euler),
            self.base_quat[envs_idx],
        )
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat
        )

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = (
            0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
        )
        self.base_ang_vel[envs_idx] = (
            0.0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
        )
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1
        )
        self.robot.set_dofs_velocity(
            velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx
        )

        self._resample_commands(envs_idx)

        # reset buffers
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        # fill extras
        self.extras['episode'] = {}
        for key in self.episode_sums.keys():
            self.extras['episode']['rew_' + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item()
                    / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        # send timeout info to the algorithm
        if self.env_cfg['send_timeouts']:
            self.extras['time_outs'] = self.time_out_buf

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
        self._prepare_reward_function()
        self._randomize_controls()
        self._randomize_rigids()
        self.tmp_dof_pos_list = []
        self.tmp_dof_vel_list = []
        self.init = True

    def perform_control(self, action):
        if not self.init:
            self.initialize()
        if action is not None and 'control' in action:
            action = action['control']
        clip_actions = self.env_cfg['clip_actions']
        if action is not None:
            self.actions = torch.clip(action, -clip_actions, clip_actions)

    def step(self, actions=None, perform_physics_step=False, robot_step=0):
        if not self.init:
            self.initialize()
        exec_actions = self.last_actions if self.action_latency > 0 else self.actions
        exec_actions = exec_actions.to(self.device)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        if robot_step == 0 or robot_step == 2:
            self.tmp_dof_pos_list.append(self.robot.get_dofs_position().detach().cpu())
            self.tmp_dof_vel_list.append(self.robot.get_dofs_velocity().detach().cpu())
        self.torques = self._compute_torques(exec_actions)
        torques = self.torques.squeeze()
        self.robot.control_dofs_force(torques, self.motor_dofs)

        if perform_physics_step:
            self.dof_pos_list = self.tmp_dof_pos_list
            self.dof_vel_list = self.tmp_dof_vel_list
            self.post_physics_step()
            self.tmp_dof_pos_list = []
            self.tmp_dof_vel_list = []

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['randomize_friction']:
            self._randomize_link_friction(env_ids)
        if self.env_cfg['randomize_contact_damping']:
            self._randomize_contact_damp_ratio(env_ids)
        if self.env_cfg['randomize_base_mass']:
            self._randomize_base_mass(env_ids)
        if self.env_cfg['randomize_com_displacement']:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):

        if self.eval:
            return

        if env_ids == None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg['continuous_push']:
            self._randomize_continuous_push(env_ids)
        if self.env_cfg['randomize_motor_strength']:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg['randomize_motor_offset']:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg['randomize_kp_scale']:
            self._randomize_kp(env_ids)
        if self.env_cfg['randomize_kd_scale']:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):

        min_friction, max_friction = self.env_cfg['friction_range']

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_friction - min_friction) + min_friction
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_contact_damp_ratio(self, env_ids):

        min_damping, max_damping = self.env_cfg['contact_damping_range']

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
                 * (max_damping - min_damping) + min_damping
        solver.set_contact_damp_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):

        min_mass, max_mass = self.env_cfg['added_mass_range']
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) \
                     * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(added_mass, [base_link_id, ], env_ids)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.env_cfg['com_displacement_range']
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
                           * (max_displacement - min_displacement) + min_displacement
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(com_displacement, [base_link_id, ], env_ids)

    def _randomize_continuous_push(self, env_ids):

        min_push, max_push = self.env_cfg['continuous_push_force_range']

        self.continuous_push[env_ids, 0] = gs.rand((len(env_ids),), dtype=float) \
                                           * (max_push - min_push) + min_push

    def _randomize_motor_strength(self, env_ids):

        min_strength, max_strength = self.env_cfg['motor_strength_range']
        self.motor_strengths[env_ids, :] = gs.rand((len(env_ids), 1), dtype=float) \
                                           * (max_strength - min_strength) + min_strength

    def _randomize_motor_offset(self, env_ids):

        min_offset, max_offset = self.env_cfg['motor_offset_range']
        self.motor_offsets[env_ids, :] = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                                         * (max_offset - min_offset) + min_offset

    def _randomize_kp(self, env_ids):

        min_scale, max_scale = self.env_cfg['kp_scale_range']
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):

        min_scale, max_scale = self.env_cfg['kd_scale_range']
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) \
                    * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]



    def gs_transform_by_quat(self, quat):
        qw, qx, qy, qz = quat.unbind(-1)
        pos = torch.tensor([1., 0., 0.]).to(quat.device)

        rot_matrix = torch.stack(
            [
                1.0 - 2 * qy ** 2 - 2 * qz ** 2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx ** 2 - 2 * qz ** 2,
                2 * qy * qz - 2 * qx * qw,
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx ** 2 - 2 * qy ** 2,
            ],
            dim=-1,
        ).reshape(*quat.shape[:-1], 3, 3)
        rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

        return rotated_pos.cpu().numpy()

    def render_ego_view(
            self,
            rotation_offset=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            depth=False,
            segmentation=False,
    ):
        head_pos = self.robot.get_pos()
        facing_direction = self.gs_transform_by_quat(self.robot.get_quat())
        head_pos = head_pos.cpu().numpy()
        head_pos += facing_direction * 0.3  # Move the camera forward by a certain distance to prevent it from seeing the avatar.

        # Compute transform on CPU (numpy) to avoid MPS torch.cross bugs on macOS
        up = np.array([0., 0., 1.], dtype=head_pos.dtype)
        T = geom_utils.pos_lookat_up_to_T(head_pos, facing_direction + head_pos, up)
        self.ego_view.set_pose(transform=T)
        rgb, depth, seg, _ = self.ego_view.render(depth=depth, segmentation=segmentation, colorize_seg=False)
        return rgb, depth, seg, self.ego_view.transform

    def _step_contact_targets(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        bounds = self.commands[:, 6]
        offsets = self.commands[:, 7]
        durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                   smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                           1 - smoothing_cdf_start(
                                       torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        measured_heights = 0.0  # no terrain
        base_height = self.base_pos[:, 2] - measured_heights
        base_height_target = self.commands[:, 9]  # + self.reward_cfg['base_height_target']
        return torch.square(base_height - base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                    torch.norm(
                        self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                        dim=-1,
                    )
                    > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(
                self.commands[:, :2] - self.base_lin_vel[:, :2]
            ),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg['tracking_sigma'])

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        first_contact = (self.feet_air_time > 0.0) * contact
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
                torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact
        return rew_airTime

    def _reward_feet_max_height(self):
        # Penalize high steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        first_contact = torch.logical_and(
            ~self.last_contacts.bool(), contact.bool()
        )
        self.last_contacts = contact
        rew_feet_max_height = torch.sum(
            torch.clamp_min(self.reward_cfg['target_feet_height'] - self.feet_max_height, 0) * ~first_contact, dim=1
        )
        feet_geom_offset = self.env_cfg['feet_geom_offset']
        for i, feet_link_idx in enumerate(self.feet_link_indices):
            feet_pos = self.robot.links[feet_link_idx].geoms[feet_geom_offset].get_pos()
            self.feet_max_height[:, i] = torch.maximum(
                self.feet_max_height[:, i], feet_pos[:, 2]
            )
        self.feet_max_height *= ~contact
        return rew_feet_max_height

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.link_contact_forces[:, self.feet_link_indices, :2], dim=2)
            > 5 * torch.abs(self.link_contact_forces[:, self.feet_link_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
                    torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_similar_to_default(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:3], dim=2).view(self.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                    torch.norm(
                        self.link_contact_forces[:, self.feet_link_indices, :], dim=-1
                    )
                    - self.reward_cfg['max_contact_force']
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = torch.sum((1 - desired_contact) * (1 - torch.exp(-foot_forces ** 2 / 100.)), dim=1) / 4
        return reward

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = torch.sum(desired_contact * (1 - torch.exp(-foot_velocities ** 2 / 10.)), dim=1) / 4
        return reward

    def _reward_feet_height(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)  # - reference_heights
        target_height = self.commands[:, 10:11] * phases + 0.02  # offset for foot radius 2cm
        # print(self.desired_contact_states[0, :].cpu(), foot_height[0, :].cpu(), torch.norm(self.link_contact_forces[:, self.feet_link_indices, :], dim=-1)[0, :].cpu())
        rew_foot_height = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_height, dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.com.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        desired_stance_width = self.commands[:, 13:14]
        desired_ys_nom = torch.cat(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            dim=1)

        desired_stance_length = self.commands[:, 14:15]
        desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2,
                                    -desired_stance_length / 2], dim=1)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.commands[:, 4]
        x_vel_des = self.commands[:, 0:1]
        y_vel_des = self.commands[:, 1:2]
        yaw_vel_des = self.commands[:, 2:3]

        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        yaw_to_y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_yaw_to_ys_offset = phases * yaw_to_y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_yaw_to_ys_offset[:, 2:4] *= -1
        yaw_to_x_vel_des = yaw_vel_des * desired_stance_width / 2
        desired_yaw_to_xs_offset = phases * yaw_to_x_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_yaw_to_xs_offset[:, [0, 2]] *= -1

        desired_ys = desired_ys_nom + (desired_ys_offset + desired_yaw_to_ys_offset)
        desired_xs = desired_xs_nom + (desired_xs_offset + desired_yaw_to_xs_offset)

        desired_footsteps_body_frame = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reward_alive(self):
        return 1

    def _reward_action_limit(self):
        # Penalize actions greater than soft_action_limit
        return torch.sum(torch.square(
            torch.abs(self.actions).clip(min=self.reward_cfg['soft_action_limit']) - self.reward_cfg[
                'soft_action_limit']), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.actions - self.last_actions)
        diff = diff * (self.last_actions != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.actions - 2 * self.last_actions + self.last_last_actions)
        diff = diff * (self.last_actions != 0) * (self.last_last_actions != 0)  # ignore first&second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, 11:13]
        quat_roll = gs_quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = gs_quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                             torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_base_quat = gs_quat_mul(quat_roll, quat_pitch)
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_unused_feet_vel(self):
        foot_velocities = torch.square(self.foot_velocities[:, :, 2])
        reward = (foot_velocities * (1 - self.commands[:, 15:19])).sum(dim=1)
        return reward

    def _reward_unused_feet_height(self):
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)
        target_height = self.commands[:, 19:20] + 0.02
        rew_foot_height = torch.square(target_height - foot_height) * (1 - self.commands[:, 15:19])
        return torch.sum(rew_foot_height, dim=1)

    def _reward_unused_feet_contact_force(self):
        foot_forces = torch.abs(self.link_contact_forces[:, self.feet_link_indices, 2])
        reward = torch.sum(1 - torch.exp(-foot_forces ** 2 / 100.), dim=1) / 4
        return reward

    def _reward_temp_action_penalty(self):
        return torch.square(self.actions[:, 3:9]).sum(dim=1)
