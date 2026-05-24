from .replay_motion_module import ReplayMotionModule
from .utils import AvatarState, ActionStatus, Mirrored_Mixamo_data, Mixamo_data_to_controller_pose
import genesis.utils.geom as geom_utils
import numpy as np
import genesis as gs

class EnterBikeMotion(ReplayMotionModule):
    def __init__(self, motion_name, motion_data, robot, name=None):
        super().__init__(motion_name, motion_data, robot, name)
        self.mirrored_data = []
        mirrored_motion_data = Mirrored_Mixamo_data(motion_data)
        for i in range(mirrored_motion_data["trans"].shape[0]):
            self.mirrored_data.append(Mixamo_data_to_controller_pose(
                mirrored_motion_data["trans"][i], mirrored_motion_data["rot"][i], mirrored_motion_data["joint"][i]
            ))

    def start(self, hand_id, bike):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        self.robot.action_state = self.motion_name
        self.robot.base_state = AvatarState.IN_VEHICLE
        self.robot.action_status = ActionStatus.ONGOING
        self.hand_id = hand_id
        self.bike = bike
        self.at_stage = 0
        self.at_frame = 0
    
    def step(self, skip_avatar_animation=False):
        data = self.data if self.hand_id == 0 else self.mirrored_data # Data left-handed
        self.at_frame += 1
        if skip_avatar_animation:
            self.at_frame = len(data) - 1
        self.robot.pose = data[self.at_frame]
        self.robot.node_trans = self.node_data[self.at_frame]
        self.robot.global_mat = self.global_mat
        self.robot.global_mat_inv = self.global_mat_inv
        if self.at_frame == len(data) - 1:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot._h_attach_to = self.bike
            self.robot._h_attach_to.occupied = True
            # Set global trans/rot to the last frame, but keep the pose
            end_trans = data[-1][:3]
            start_trans = data[0][:3]
            delta_trans = end_trans - start_trans
            delta_trans[1] = 0 # Only use xz-translation in motion to refine global_trans
            self.robot.global_trans = self.robot.global_trans + self.robot.global_rot @ self.robot.base_rot @ delta_trans
            self.robot.pose[:3] = end_trans - delta_trans
            end_rot = geom_utils.quat_to_R(data[-1][3:7])
            start_rot = geom_utils.quat_to_R(data[0][3:7])
            delta_rot = end_rot @ np.linalg.inv(start_rot)
            delta_rot_z = delta_rot @ np.array([0.0, 0.0, 1.0])
            delta_rot_z[1] = 0
            delta_rot_z = delta_rot_z / np.linalg.norm(delta_rot_z) # Only use y-rotation on z-axis in motion to refine global_rot
            delta_rot = np.array([
                [delta_rot_z[2], 0.0, delta_rot_z[0]],
                [0.0, 1.0, 0.0],
                [-delta_rot_z[0], 0.0, delta_rot_z[2]],
            ], dtype=delta_rot.dtype)
            self.robot.global_rot = self.robot.global_rot @ self.robot.base_rot @ delta_rot @ np.linalg.inv(self.robot.base_rot)
            self.robot.pose[3:7] = geom_utils.R_to_quat(np.linalg.inv(delta_rot) @ end_rot)