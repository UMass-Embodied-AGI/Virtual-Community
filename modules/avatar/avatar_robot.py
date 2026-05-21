import os
import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
from .utils import *
import sys
sys.path.insert(0, os.getcwd())
from tools.utils import get_assets_dir

from scipy.spatial.transform import Rotation as R

class AvatarRobot:
    def __init__(self, env, skin_options, name):
        mat_avatar = gs.materials.Avatar()
        self.box = env.add_entity(
            type="avatar_box",
            name=name,
            material=gs.materials.Rigid(),
            morph=gs.morphs.Box(
                lower=(-0.3, -0.3, 0.5),
                upper=( 0.3,  0.3, 1.5),
                fixed=False,
                visualization=False,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0),
            ),
        )
        self.joint_num = SMPLX_JOINT_NUM

        if skin_options is not None:
            glb_path = skin_options['glb_path']
            if not os.path.isabs(glb_path):
                glb_path = os.path.join(get_assets_dir(), glb_path)
            self.skin = env.add_entity(
                type="avatar",
                name=name,
                material=mat_avatar,
                morph=gs.morphs.Mesh(
                    file=glb_path,
                    euler=skin_options['euler'],
                    pos=skin_options['pos'],
                    decimate=False,
                    convexify=False,
                    collision=False,
                    group_by_material=False,
                    # euler=[0,0,0] # for static skin
                )
            )
            self.skin_rot = geom_utils.euler_to_R(skin_options['euler'])
        else:
            self.skin = None

        self.base_state = AvatarState.NO_STATE
        self.action_state = AvatarState.NO_ACTION
        self.action_status = ActionStatus.INIT
        self.attached_object = [None, None]

        self.base_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.global_trans = np.zeros(3, dtype=np.float64)
        self.global_rot = np.eye(3, dtype=np.float64)
        self.pose = np.zeros(7 + self.joint_num * 4)
        self.stop_pose = np.zeros_like(self.pose)
        self.stop_node = []
        self.stop_mat = np.zeros((65, 3, 3), dtype=np.float64)
        self.stop_mat_inv = np.zeros((65, 3, 3), dtype=np.float64)
        self.sit_mat = np.zeros((65, 3, 3), dtype=np.float64)
        self.sit_mat_inv = np.zeros((65, 3, 3), dtype=np.float64)
        self.global_mat = np.zeros_like(self.stop_mat)
        self.sit_pose = np.zeros_like(self.pose)
        self.sit_node = []

        self._h_attach_to = None
        
        self.no_collision_idx = []

    def reset(
        self,
        global_trans: np.ndarray = np.zeros(3, dtype=np.float64),
        global_rot: np.ndarray = np.eye(3, dtype=np.float64),
    ):
        self.base_state = AvatarState.STANDING
        self.action_state = AvatarState.NO_ACTION
        self.action_status = ActionStatus.INIT
        self.attached_object = [None, None]

        self.global_trans = global_trans
        self.global_rot = global_rot
        self.pose = self.stop_pose
        self.node_trans = self.stop_node
        self.global_mat = self.stop_mat
        self.global_mat_inv = self.stop_mat_inv

        self._h_attach_to = None

        self.update()
    
    def initialize_no_collision(
        self,
        entities,
    ):
        self.no_collision_idx = []
        for t in entities:
            for geom in t.geoms:
                self.no_collision_idx.append(geom.idx)
    
    def add_no_collision(
        self,
        entity,
    ):
        for geom in entity.geoms:
            self.no_collision_idx.append(geom.idx)
    
    def del_no_collision(
        self,
        entity,
    ):
        for geom in entity.geoms:
            self.no_collision_idx.remove(geom.idx)
    

    def get_global_xy(
        self,
    ):
        if self.action_state == "walk":
            total_trans = self.global_rot @ self.base_rot @ self.pose[:3] + self.global_trans
        else:
            total_trans = self.global_trans.copy()
        return total_trans[0], total_trans[1]
    
    def get_global_height(
        self,
    ):
        return self.global_trans[2]
    
    def get_global_pose(
        self,
    ):
        # The real global xy (w/ motion) and the rough rot (w/o motion)
        x, y = self.get_global_xy()
        z = self.get_global_height()
        ypr = geom_utils.R_to_ypr(self.global_rot)
        return np.array([x, y, z, ypr[2], ypr[1], ypr[0]], dtype=self.global_trans.dtype)
    
    def update(self):
        # Apply motion transformation -> Apply base rotation -> Apply global transformation
        motion_trans = self.pose[:3]
        motion_rot = geom_utils.quat_to_R(self.pose[3:7])
        total_trans = self.global_rot @ self.base_rot @ motion_trans + self.global_trans
        self.box.set_pos(self.get_global_pose()[:3] + np.array([0, 0, 0.959008030]))
        self.box.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        if self.skin is not None:
            skin_base_rot = geom_utils.R_to_quat(geom_utils.euler_to_R(
                R.from_matrix(self.global_rot).as_euler('xyz', degrees=True)[[0,2,1]]) @ motion_rot)
            
            self.skin.update_mesh(
                total_trans[[1,2,0]],
                self.global_mat, self.global_mat_inv, skin_base_rot,
                self.node_trans
            )

        if self.attached_object[0] is not None:
            Pinky_pos = self.skin.get_global_translation("LeftHandPinky4")[0]
            
            Thumb_pos = self.skin.get_global_translation("LeftHandThumb4")[0]
            
            hand_pos = self.skin.get_global_translation("LeftHand")[0]

            rot = np.zeros((3, 3))
            rot[:,0] = Thumb_pos - Pinky_pos
            rot[:,0] = rot[:,0] / np.linalg.norm(rot[:,0])
            rot[:,1] = hand_pos - Thumb_pos
            rot[:,1] -= (rot[:,1] @ rot[:,0]) * rot[:,0]
            rot[:,1] = rot[:,1] / np.linalg.norm(rot[:,1])
            rot[:,2] = np.cross(rot[:,0], rot[:,1])
            
            obj_pos = Pinky_pos

            self.attached_object[0].set_pos(obj_pos)
            self.attached_object[0].set_quat(geom_utils.R_to_quat(rot))
        
        if self.attached_object[1] is not None:
            Pinky_pos = self.skin.get_global_translation("RightHandPinky4")[0]

            Thumb_pos = self.skin.get_global_translation("RightHandThumb4")[0]
            
            hand_pos = self.skin.get_global_translation("RightHand")[0]

            rot = np.zeros((3, 3))
            rot[:,0] = Thumb_pos - Pinky_pos
            rot[:,0] = rot[:,0] / np.linalg.norm(rot[:,0])
            rot[:,1] = hand_pos - Thumb_pos
            rot[:,1] -= (rot[:,1] @ rot[:,0]) * rot[:,0]
            rot[:,1] = rot[:,1] / np.linalg.norm(rot[:,1])
            rot[:,2] = -np.cross(rot[:,0], rot[:,1])
            
            obj_pos = Pinky_pos

            self.attached_object[1].set_pos(obj_pos)
            self.attached_object[1].set_quat(geom_utils.R_to_quat(rot))

    def get_hand_pos(
        self,
        hand_id,
    ):
        if hand_id == 0:
            return self.skin.get_global_translation("LeftHand")[0]
        else:
            return self.skin.get_global_translation("RightHand")[0]
      