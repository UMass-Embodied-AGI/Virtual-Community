import numpy as np
from .base_motion_module import BaseMotionModule
from .utils import AvatarState, ActionStatus
from scipy.spatial.transform import Rotation as R, Slerp
from .utils import Mixamo_global_processing
import genesis.utils.geom as geom_utils
import genesis as gs

class ReachObjMotion(BaseMotionModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = 0        
        self.total_rotate_frames = 5 
        self.total_reach_frames = 20  
        self.total_return_frames = 20 
        self.current_pose_cache = {}
        self.target_pose_cache = {} 
        self.initial_global_rot = None
        self.final_global_rot = None

    def start(self, hand_id, obj, obj_pos=None):
        if self.robot.action_state != AvatarState.NO_ACTION:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: AvatarState is {self.robot.action_state}.")
            return
        if self.robot.base_state != AvatarState.STANDING:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: BaseState is {self.robot.base_state}")
            return
        if self.robot.attached_object[hand_id] is not None:
            gs.logger.warning(f"Cannot start motion {self.motion_name}: hand {hand_id} is already holding an object.")
            return

        self.robot.action_state = self.motion_name
        self.robot.action_status = ActionStatus.ONGOING
        
        self.hand_id = hand_id
        self.target_obj = obj
        
        self.at_frame = 0
        self.phase = 0

        self.initial_global_rot = self.robot.global_rot
        orientation_vec = (self.target_obj.get_pos().cpu() - self.robot.global_trans) * np.array([1.0, 1.0, 0.0])
        orientation_vec /= np.linalg.norm(orientation_vec)

        self.final_global_rot = np.array([
                [orientation_vec[0], -orientation_vec[1], 0],
                [orientation_vec[1],  orientation_vec[0], 0],
                [0,                   0,                  1]
            ], dtype=float)
        
        self.slerp = Slerp([0, 1], R.from_matrix([self.initial_global_rot, self.final_global_rot]))
        self.hand_pos = self.robot.get_hand_pos(self.hand_id)
        if obj_pos is not None:
            self.obj_pos = obj_pos
        else:
            self.obj_pos = self.target_obj.get_pos().cpu().numpy()
        
        self.cache_node_trans = []
        self.robot.add_no_collision(obj)

    def step(self, skip_avatar_animation=False):
        if self.robot.action_state != self.motion_name:
            return
        
        if skip_avatar_animation:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.attached_object[self.hand_id] = self.target_obj
            self.robot.pose = self.robot.stop_pose
            self.robot.node_trans = self.robot.stop_node
            self.robot.global_mat = self.robot.stop_mat
            self.robot.global_mat_inv = self.robot.stop_mat_inv
            return
        
        if self.phase == 0:
            alpha = self.at_frame / float(self.total_rotate_frames)
            self.robot.global_rot = self.slerp([alpha])[0].as_matrix()

            self.at_frame += 1
            if self.at_frame >= self.total_rotate_frames:
                self.phase = 1
                self.at_frame = 0
                
                pose = self.robot.stop_pose
                total_trans = self.robot.global_rot @ self.robot.base_rot @ pose[:3] + self.robot.global_trans
                skin_base_rot = geom_utils.R_to_quat(
                    geom_utils.euler_to_R(
                        R.from_matrix(self.robot.global_rot).as_euler('xyz', degrees=True)[[0,2,1]]) @ geom_utils.quat_to_R(pose[3:7])
                )
                
                self.robot.skin.calculate_real_pos(
                    rotation=Mixamo_global_processing(pose[3:].reshape(-1,4), self.robot.global_mat, self.robot.global_mat_inv, skin_base_rot),
                    base_translation=total_trans[[1,2,0]]
                )


        elif self.phase == 1:
            alpha = self.at_frame / float(self.total_reach_frames)
            
            desired_pos = self.hand_pos + alpha * (self.obj_pos - self.hand_pos)
            # import pdb
            # pdb.set_trace()
            if self.hand_id == 0:
                # self.robot.node_trans = self.robot.skin.ik_solve("LeftHand", "LeftShoulder", desired_pos)
                self.robot.node_trans = self.robot.skin.ik_solve("LeftHand", "LeftShoulder", desired_pos)
            else:
                # self.robot.node_trans = self.robot.skin.ik_solve("RightHand", "RightShoulder", desired_pos)
                self.robot.node_trans = self.robot.skin.ik_solve("RightHand", "RightShoulder", desired_pos)
            self.at_frame += 1
            self.cache_node_trans.append(self.robot.node_trans)
            if self.at_frame > self.total_reach_frames:
                self.phase = 2
                self.at_frame = 0

        elif self.phase == 2:
            self.robot.attached_object[self.hand_id] = self.target_obj
            self.at_frame += 1
            if self.at_frame >= 5: 
                self.phase = 3
                self.at_frame = 1
        elif self.phase == 3:
            self.robot.node_trans = self.cache_node_trans[-self.at_frame]
            self.at_frame += 1
            if self.at_frame >= self.total_return_frames:
                self.phase = 4
                self.at_frame = 0
        elif self.phase == 4:
            self.robot.action_state = AvatarState.NO_ACTION
            self.robot.action_status = ActionStatus.SUCCEED
            self.robot.pose = self.robot.stop_pose
            self.robot.node_trans = self.robot.stop_node
            self.robot.global_mat = self.robot.stop_mat
            self.robot.global_mat_inv = self.robot.stop_mat_inv
