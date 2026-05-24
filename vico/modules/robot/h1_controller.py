import torch
from .go2_controller import Go2Controller
from .robot_utils import (gs_rand_float, gs_quat_apply_yaw, gs_quat_conjugate, gs_quat_from_angle_axis,
                          gs_quat_mul, gs_inv_quat, gs_transform_by_quat)
import numpy as np


class H1Controller(Go2Controller):
    def _prepare_obs_noise(self):
        self.obs_noise[10:13] = self.obs_cfg['obs_noise']['ang_vel']
        self.obs_noise[16:26] = self.obs_cfg['obs_noise']['dof_pos']
        self.obs_noise[26:36] = self.obs_cfg['obs_noise']['dof_vel']
        self.obs_noise[36:39] = self.obs_cfg['obs_noise']['gravity']

    def compute_observations(self):
        self.obs_buf = torch.cat(
            [
                self.actions,  # 10
                self.base_ang_vel * self.obs_scales['ang_vel'],  # 3
                self.commands[:, 2:3] * self.obs_scales['ang_vel'],  # 1
                self.commands[:, :2] * self.obs_scales['lin_vel'],  # 2
                (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],  # 10
                self.dof_vel * self.obs_scales['dof_vel'],  # 10
                self.projected_gravity,  # 3
                self.clock_inputs,  # 2
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
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],  # 10
                    self.dof_vel * self.obs_scales['dof_vel'],  # 10
                    self.actions,  # 10
                    self.last_actions,  # 10
                    self.commands[:, 4:],  # 8
                    self.clock_inputs,  # 2
                ],
                axis=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

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
            self.commands[envs_idx, 2] *= self.commands[envs_idx, 2] > 0.2

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
            if self.command_cfg['gaits'][i] == 'jump':
                self.commands[category_envs_idx, 5] = 0.
            elif self.command_cfg['gaits'][i] == 'walk':
                self.commands[category_envs_idx, 5] = 0.5

        # gait duration
        self.commands[envs_idx, 6] = gs_rand_float(
            *self.command_cfg['gait_duration_range'], (len(envs_idx),), self.device
        )

        # body height
        self.commands[envs_idx, 7] = gs_rand_float(
            *self.command_cfg['body_height_range'], (len(envs_idx),), self.device
        )

        # foot swing height
        self.commands[envs_idx, 8] = gs_rand_float(
            *self.command_cfg['foot_swing_height_range'], (len(envs_idx),), self.device
        )
        # self.commands[envs_idx, 10] *= self.commands[envs_idx, 10] > 0.01
        self.commands[envs_idx, 8] = torch.clip(self.commands[envs_idx, 10], max=self.commands[envs_idx, 9] - 0.15)

        # body pitch & roll
        self.commands[envs_idx, 9] = gs_rand_float(
            *self.command_cfg['body_pitch_range'], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 10] = gs_rand_float(
            *self.command_cfg['body_roll_range'], (len(envs_idx),), self.device
        )

        # stance width & length
        self.commands[envs_idx, 11] = gs_rand_float(
            *self.command_cfg['stance_width_range'], (len(envs_idx),), self.device
        )

    def _step_contact_targets(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        durations = self.commands[:, 6]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases,
                        self.gait_indices + 0, ]  # DO NOT REMOVE + 0

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_L = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                  smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                          1 - smoothing_cdf_start(
                                      torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_R = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                  smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                          1 - smoothing_cdf_start(
                                      torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_L
        self.desired_contact_states[:, 1] = smoothing_multiplier_R

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.com.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 2, 3, device=self.device)
        for i in range(2):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply_yaw(gs_quat_conjugate(self.base_quat),
                                                                 cur_footsteps_translated[:, i, :])

        desired_stance_width = self.commands[:, 11:12]
        desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, ], dim=1)
        desired_xs_nom = torch.zeros_like(desired_ys_nom, device=desired_ys_nom.device, dtype=desired_ys_nom.dtype)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.commands[:, 4]
        x_vel_des = self.commands[:, 0:1]
        y_vel_des = self.commands[:, 1:2]
        yaw_vel_des = self.commands[:, 2:3]

        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        yaw_to_x_vel_des = yaw_vel_des * desired_stance_width / 2
        desired_yaw_to_xs_offset = phases * yaw_to_x_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_yaw_to_xs_offset[:, 0] *= -1

        desired_ys = desired_ys_nom + desired_ys_offset
        desired_xs = desired_xs_nom + (desired_xs_offset + desired_yaw_to_xs_offset)

        desired_footsteps_body_frame = torch.cat((desired_xs.unsqueeze(2), desired_ys.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        measured_heights = self.terrain_heights
        base_height = self.base_pos[:, 2] - measured_heights
        base_height_target = self.commands[:, 7]  # + self.reward_cfg['base_height_target']
        return torch.square(base_height - base_height_target)

    def _reward_feet_height(self):
        reference_heights = self.terrain_heights[:, None]
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - reference_heights
        return torch.sum(foot_height.clamp(max=0.25) * phases, dim=1)

    def _reward_contact_force(self):
        contact_force = torch.max(self.link_contact_forces[:, self.feet_link_indices, 2], dim=1)[0]
        rew_contact_force = torch.square((400 - contact_force).clamp(min=0) / 400)
        return rew_contact_force

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.commands[:, 9:11]
        quat_roll = gs_quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                            torch.tensor([1, 0, 0], device=self.device, dtype=torch.float))
        quat_pitch = gs_quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                             torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))

        desired_rot_quat = gs_quat_mul(quat_roll, quat_pitch)
        desired_base_quat = gs_quat_mul(desired_rot_quat, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_feet_max_height(self):
        # Penalize high steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts_filt = contact_filt
        first_contact = torch.logical_and(
            ~self.last_contacts_filt.bool(), contact_filt.bool()
        )
        rew_feet_max_height = torch.sum(
            torch.clamp_min(self.commands[:, None, 8:9].repeat(1, 2, 1) - self.feet_max_height, 0) * ~first_contact,
            dim=1
        )
        self.feet_max_height = torch.max(self.feet_max_height, self.foot_positions[..., 2])
        self.feet_max_height *= ~contact
        return rew_feet_max_height

    def _reward_feet_align(self):
        left_quat = self.foot_quaternions[:, 0]
        left_gravity = gs_transform_by_quat(self.global_gravity, gs_inv_quat(left_quat))
        right_quat = self.foot_quaternions[:, 1]
        right_gravity = gs_transform_by_quat(self.global_gravity, gs_inv_quat(right_quat))
        return torch.square(left_gravity - right_gravity).sum(dim=-1)

    def _reward_feet_slip(self):
        contact = self.foot_positions[..., 2] < 0.15
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact * foot_velocities, dim=1)
        return rew_slip

    def _reward_hip_yaw(self):
        reward = (
                torch.square(self.actions[:, 0]) +
                torch.square(self.actions[:, 5])
        )
        return reward

    def _reward_hip_roll(self):
        reward = (
                torch.square(self.actions[:, 1]) +
                torch.square(self.actions[:, 6])
        )
        return reward
