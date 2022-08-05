import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper


def forward_kinematics(robot, q, frame_id):
    # calculates the position of a given frame at a given configuration q
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pose = pin.updateFramePlacement(robot.model, robot.data, frame_id)
    return pose.translation

def jacobian(robot, q, frame_id):
    # calculates the derivative of the forward kinematics function of a given frame at a given configuration q
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pose = pin.updateFramePlacement(robot.model, robot.data, frame_id)
    body_jacobian = pin.computeFrameJacobian(robot.model, robot.data, q, frame_id)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = pose.rotation
    Ad[3:, 3:] = pose.rotation
    J = Ad @ body_jacobian
    Jlin = J[:3, :]

    return Jlin

def pinv(J, eps=1e-3):
    # calculates the right pseudoinverse of a matrix J
    JJT = J @ J.T
    return J.T @ np.linalg.inv((JJT + eps * np.eye(*JJT.shape)))

def inverse_kinematics(robot, p_des, frame_id, q0, max_it=200, alpha=0.1, eps=0.01):
    q = q0
    
    for i in range(max_it):
        p = forward_kinematics(robot, q, frame_id)
        err = np.linalg.norm(p - p_des)
        if err <= eps:
            return q
        J = jacobian(robot, q, frame_id)
        q = q - alpha * pinv(J) @ (forward_kinematics(robot, q, frame_id) - p_des)

    return False

def findAngle(p_des, q0, robot, frame_id):
    q = inverse_kinematics(robot, p_des, frame_id, q0)
    if isinstance(q, bool) and not q:
#         print('inverse kinematics failed for position: ', p_des)
        return False
    return q 

def pinv_left(J):
    return np.linalg.pinv(J)


def inverse_kinematics_two_frames(robot, p_des1, p_des2, 
                                  frame_id1, frame_id2, q0, 
                                  eps = 0.10, eps1 = 0.10, eps2 = 0.20, max_it = 200, alpha = 0.1):
    q = q0

    for i in range(max_it):
        
        J1 = jacobian(robot, q, frame_id1)
        J2 = jacobian(robot, q, frame_id2)
        J = np.vstack((J1, J2))
        
        p_des = np.concatenate((p_des1, p_des2))
        
        p1 = forward_kinematics(robot, q, frame_id1)
        p2 = forward_kinematics(robot, q, frame_id2)
        p = np.concatenate((p1, p2))
        
        err1 = np.linalg.norm(p1 - p_des1)
        err2 = np.linalg.norm(p2 - p_des2)
        err = np.linalg.norm(p - p_des)
        if err <= eps:
            return q
        
        q = q - alpha * pinv_left(J) @ (p - p_des)
        
    if err1 <= eps1 and err2 <= eps2:
        return q
    
    return False


def findAngleTwoFrames(robot, p_des1, p_des2, 
                              frame_id1, frame_id2, q0, 
                              eps=0.10, eps1=0.10, eps2=0.20):
    q = inverse_kinematics_two_frames(robot, p_des1, p_des2, 
                                      frame_id1, frame_id2, q0, 
                                      eps, eps1, eps2)
    if isinstance(q, bool) and not q:
#         print('inverse kinematics failed for position: ', np.concatenate((p_des_wrist, p_des_elbow)))
        return False
    else:
        return q 