import numpy as np

def quat2rotmat(quat):
    """
    Convert a quaternion ([w, x, y, z]) to a 3x3 rotation matrix.

    Args:
    quat: A length-4 array in [w, x, y, z] format (must be normalized)

    Returns:
    R: 3x3 rotation matrix
    """
    w, x, y, z = quat
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def euler2rotmat(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a 3x3 rotation matrix.
    Rotation order: Z-Y-X (first rotate about Z by yaw, then about Y by pitch, and finally about X by roll), consistent with the formula provided by the user.

    Args:
    roll: Rotation angle about the X-axis (radians)
    pitch: Rotation angle about the Y-axis (radians)
    yaw: Rotation angle about the Z-axis (radians)

    Returns:
    R: 3x3 rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])
    return R

def transform2mat(x, y, z, roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr, x],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr, y],
        [-sp,   cp*sr,            cp*cr,            z],
        [0,     0,                0,                1]
    ])

def mat2transform(mat):
    x, y, z = mat[0:3, 3]
    roll, pitch, yaw = np.arctan2(mat[2, 1], mat[2, 2]), np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2)), np.arctan2(mat[1, 0], mat[0, 0])
    return x, y, z, roll, pitch, yaw

def euler2quat(roll, pitch, yaw):
    """
    [roll, pitch, yaw] to ([w, x, y, z])    
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = -cy * sp * sr + sy * cp * cr
    return np.array([w, x, y, z])

def quat2euler(quat):
    """
    Input:
        quat: [w, x, y, z]
    return:
        roll pitch yaw
    """
    w, x, y, z = map(float, quat)
    sin_pitch = 2 * (w * y - x * z)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    if np.abs(sin_pitch) < 0.9999999:
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(sin_pitch)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    else:
        pitch = np.pi / 2 if sin_pitch > 0 else -np.pi / 2
        yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (x**2 + z**2))
        roll = 0.0  
    return roll, pitch, yaw

def dampedPinv(J, lambda_d=0.1):
    J_T = J.T
    damping = lambda_d ** 2 * np.eye(J.shape[0])
    J_pinv_damped = np.dot(J_T, np.linalg.inv(np.dot(J, J_T) + damping))
    return J_pinv_damped