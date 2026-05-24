import numpy as np
from enum import Enum
import genesis.utils.geom as geom_utils
from scipy.spatial.transform import Rotation
import genesis as gs


SMPLX_JOINT_NUM = 54
MIXAMO_JOINT_NUM = 67
MIXAMO_SKIN_JOINT_NUM = 65
'''
Motion pose format:
    translation introduced by motion (3)
    rotation introduced by motion a.k.a. pelvis pose (4)
    joint pose (SMPLX_JOINT_NUM * 4)
'''

# AvatarState is for robot use (private), only implies which spare state and action is in
class AvatarState(Enum):
    # base_state: These indicate when spare, what base state the avatar is in
    NO_STATE = "BASE_NO_STATE"
    STANDING = "BASE_STANDING"
    SITTING = "BASE_SITTING"
    SLEEPING = "BASE_SLEEPING"
    IN_VEHICLE = "BASE_INVEHICLE"
    # action_state: if the robot is in action, other actions are identified by action_name
    NO_ACTION = "NO_ACTION"
    KEEPING = "KEEPING"

# ActionStatus is for outer use (public), only implies if action is executing correctly
class ActionStatus(Enum):
    INIT = "INIT"
    ONGOING = "ONGOING"
    SUCCEED = "SUCCEED"
    COLLIDE = "COLLIDE"
    FAILED_REACHING = "FAILED_REACHING"
    FAIL = "FAIL"


def transform_quat_by_quat(u, v):
    return np.array([
        u[3] * v[0] + u[0] * v[3] + u[1] * v[2] - u[2] * v[1],
        u[3] * v[1] - u[0] * v[2] + u[1] * v[3] + u[2] * v[0],
        u[3] * v[2] + u[0] * v[1] - u[1] * v[0] + u[2] * v[3],
        u[3] * v[3] - u[0] * v[0] - u[1] * v[1] - u[2] * v[2],
    ])

def parse_matrix(nodes, node_index, dummy_matrix):
    node = nodes[node_index]
    if node.matrix is not None:
        matrix = np.array(node.matrix, dtype=float).reshape((4, 4))
        assert 0, "should not use matrix"
    else:
        matrix = dummy_matrix[node_index][0]
        if node.translation is not None:
            matrix[3, :3] = node.translation
        if node.rotation is not None:
            rotation_matrix = dummy_matrix[node_index][1]
            rotation_matrix[:3, :3] = Rotation.from_quat(node.rotation).as_matrix().T
            matrix = rotation_matrix @ matrix
        if node.scale is not None:
            scale = np.array(node.scale, dtype=float)
            scale_matrix = np.diag(np.append(scale, 1))
            matrix = scale_matrix @ matrix
    return matrix

def forward_kinematics(nodes, node_index, root_matrix=None, dummy_matrix=None, root_id = None):
    matrix_list = list()
    node = nodes[node_index]
    if root_id is None:
        matrix = parse_matrix(nodes, node_index, dummy_matrix)
    else:
        matrix = dummy_matrix[node_index][0]

    if node_index == root_id:
        root_id = None
    
    if root_matrix is not None:
        matrix = matrix @ root_matrix
    
    matrix_list.append([node_index, matrix])
    
    for sub_node_index in node.children:
        sub_matrix_list = forward_kinematics(nodes, sub_node_index, matrix, dummy_matrix, root_id)
        matrix_list.extend(sub_matrix_list)
    
    return matrix_list

def preprocess_joint_T(geom, rotation):
    MAPPING = [0, 1, 2, 3, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]
    
    if len(geom._skin_joints) > 65:
        for i,r in enumerate(rotation[:65]):
            joint_id = geom._skin_joints[MAPPING[i]]
            geom._nodes[1][joint_id].rotation = \
                transform_quat_by_quat(
                    geom._init_nodes[1][joint_id].rotation if geom._init_nodes[1][joint_id].rotation is not None else [0,0,0,1],
                    r
                )
    else:
        for i,j in enumerate(geom._skin_joints):
            geom._nodes[1][j].rotation = \
                transform_quat_by_quat(
                    geom._init_nodes[1][j].rotation if geom._init_nodes[1][j].rotation is not None else [0,0,0,1],
                    rotation[i]
                )

    # forward joints
    dummy_matrixs = np.identity(4)[None].repeat(3, axis=0)[None].repeat(100, axis=0)
    return np.asarray(
        [it[1] for it in sorted(
            forward_kinematics(geom._nodes[1], geom._nodes[0], None, dummy_matrixs, geom._skin_joints[0]), 
            key=lambda x:x[0])
        ]
    )

def Mixamo_global_processing(pose: np.ndarray, global_mat: np.ndarray, global_mat_inv: np.ndarray, base_rot: np.ndarray):
    '''Input a (joint num, 4) pose array and transform to mixamo order (joint num * 4)'''
    mixamo_pose = np.array([[0.0, 0.0, 0.0, 1.0]] * (MIXAMO_JOINT_NUM))
    global_quat = np.concatenate([base_rot.reshape(1, -1), pose[1: MIXAMO_SKIN_JOINT_NUM]])
    result = np.matmul(np.matmul(global_mat_inv, geom_utils.quat_to_R(global_quat)), global_mat)
    mixamo_pose[:MIXAMO_SKIN_JOINT_NUM] = geom_utils.R_to_quat(result)[:,[1,2,3,0]]
    return mixamo_pose

def Mixamo_node_processing(geom, pose: np.ndarray, global_mat: np.ndarray, global_mat_inv: np.ndarray):
    mixamo_pose = np.array([[0.0, 0.0, 0.0, 1.0]] * (MIXAMO_JOINT_NUM))
    global_quat = pose[3:].reshape(-1,4)
    result = np.matmul(np.matmul(global_mat_inv, geom_utils.quat_to_R(global_quat)), global_mat)
    mixamo_pose[:MIXAMO_SKIN_JOINT_NUM] = geom_utils.R_to_quat(result)[:,[1,2,3,0]]
    node_transform = preprocess_joint_T(geom, mixamo_pose)
    return node_transform

def Global_quat_to_mixamo_quat(global_quat, global_mat, global_mat_inv):
    rot = global_mat_inv @ geom_utils.quat_to_R(global_quat) @ global_mat
    return geom_utils.R_to_quat(rot)[[1,2,3,0]] # [w,x,y,z] to [x,y,z,w]

def Mixamo_data_to_controller_pose(motion_trans, motion_rot, mixamo_pose):
    pose = np.concatenate([motion_trans, motion_rot, mixamo_pose.reshape(-1)])
    return pose

def Mirrored_Mixamo_data(data):
    mir_data = np.concatenate(
        [data["joint"][:,:6], data["joint"][:,30:54], data["joint"][:,6:30], data["joint"][:,59:64], data["joint"][:,54:59]],
        axis=1,
    )
    mir_data[:,:,2:] *= -1
    mir_rot = data["rot"].copy()
    mir_rot[:,2:] *= -1
    mir_trans = data["trans"].copy()
    mir_trans[:,0] *= -1
    return {
        "trans": mir_trans,
        "rot": mir_rot,
        "joint": mir_data,
    }