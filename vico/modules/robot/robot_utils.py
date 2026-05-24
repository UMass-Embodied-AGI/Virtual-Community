import numpy as np
import torch


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def gs_inv_quat(quat):
    qw, qx, qy, qz = quat.unbind(-1)
    inv_quat = torch.stack([1.0 * qw, -qx, -qy, -qz], dim=-1)
    return inv_quat


def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos


def gs_quat2euler(quat):  # xyz
    # Extract quaternion components
    qw, qx, qy, qz = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def gs_euler2quat(xyz):  # xyz

    roll, pitch, yaw = xyz.unbind(-1)

    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()

    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def gs_quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def gs_quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def gs_quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return gs_quat_apply(quat_yaw, vec)


def gs_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, :1], -a[:, 1:], ), dim=-1).view(shape)

def normalize_quat(q):
    """
    Normalize a quaternion to unit length.
    Args:
        q (np.ndarray): Quaternion [w, x, y, z].
    Returns:
        np.ndarray: Normalized quaternion.
    """
    return q / np.linalg.norm(q)

def quat_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    Args:
        q (np.ndarray): Quaternion [w, x, y, z].
    Returns:
        np.ndarray: Conjugated quaternion [w, -x, -y, -z].
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_multiply(q1, q2):
    """
    Multiply two quaternions (Hamilton product).
    Args:
        q1, q2 (np.ndarray): Quaternions [w, x, y, z].
    Returns:
        np.ndarray: Product quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

def quat_to_rot_matrix(q):
    """
    Convert a normalized quaternion to a 3x3 rotation matrix.
    Args:
        q (np.ndarray): Quaternion [w, x, y, z].
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])


def relative_transform_matrix(q1, q2):
    """
    Compute the 4x4 homogeneous transform matrix mapping orientation q1 to q2.
    That is, T such that R_rel = R2 * R1^T and T = [[R_rel, 0],[0,1]].
    Args:
        q1, q2 (array-like): Input quaternions [w, x, y, z].
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    """
    # Normalize inputs
    q1 = normalize_quat(np.asarray(q1))
    q2 = normalize_quat(np.asarray(q2))

    # Compute relative quaternion: q_rel = q2 * conj(q1)
    q_rel = quat_multiply(q2, quat_conjugate(q1))
    q_rel = normalize_quat(q_rel)

    # Convert to rotation matrix
    R_rel = quat_to_rot_matrix(q_rel)

    # Build homogeneous transform
    T = np.eye(4)
    T[:3, :3] = R_rel
    return T

def quat_to_direction(q, v_local=np.array([1.0, 0.0, 0.0])):
    """
    Rotate a local direction vector v_local by quaternion q to obtain the world-frame direction.

    Args:
        q (array-like): Quaternion [w, x, y, z], must be unit-norm.
        v_local (np.ndarray): 3-vector in the local frame (default is forward=[1,0,0]).

    Returns:
        np.ndarray: Rotated 3-vector in world frame.
    """
    w, x, y, z = q
    # build rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])
    # apply rotation
    return R.dot(v_local)
