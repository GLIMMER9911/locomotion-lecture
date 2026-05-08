"""
double pendulum trajectory tracking control with MuJoCo.
Dynamics is given 
Run :
    python doublependulum_adaptive_control.py
"""
import os
import sys
import numpy as np
from pathlib import Path
from src.mujoco_viewer import MuJoCoSim
from src.double_pendulum import DoublePendulum
from src.plotting import plot_joint_trajectories, plot_joint_trajectory_comparison_subplots

def joint_trajectory(t):

    q_ref = np.array([
        np.pi / 8 - np.pi / 8 * np.cos(t),
        0
    ])

    dq_ref = np.array([
        np.pi / 8 * np.sin(t),
        0
    ])  

    dqq_ref = np.array([
        np.pi / 8 * np.cos(t),
        0
    ])
    return q_ref, dq_ref, dqq_ref

def joint_trajectory_rich(t):
    q_ref = np.array([
        np.pi / 8 - np.pi / 8 * np.cos(t),
        np.pi / 10 * np.sin(1.5 * t)
    ])

    dq_ref = np.array([
        np.pi / 8 * np.sin(t),
        np.pi / 10 * 1.5 * np.cos(1.5 * t)
    ])

    ddq_ref = np.array([
        np.pi / 8 * np.cos(t),
        -np.pi / 10 * 1.5**2 * np.sin(1.5 * t)
    ])

    return q_ref, dq_ref, ddq_ref

def regressor_matrix(q, dq, dqr, ddqr, l1=1.0, l2=1.0, g=9.81):
    q1, q2 = q
    dq1, dq2 = dq
    dqr1, dqr2 = dqr
    ddqr1, ddqr2 = ddqr

    delta = q2 - q1

    Y = np.zeros((2, 2))

    # tau_1 coefficient of m1
    Y[0, 0] = l1**2 * ddqr1 + g * l1 * np.sin(q1)

    # tau_1 coefficient of m2
    Y[0, 1] = (
        l1**2 * ddqr1
        + l1 * l2 * np.cos(delta) * ddqr2
        - l1 * l2 * np.sin(delta) * dq2 * dqr2
        + g * l1 * np.sin(q1)
    )

    # tau_2 coefficient of m1
    Y[1, 0] = 0.0

    # tau_2 coefficient of m2
    Y[1, 1] = (
        l1 * l2 * np.cos(delta) * ddqr1
        + l2**2 * ddqr2
        + l1 * l2 * np.sin(delta) * dq1 * dqr1
        + g * l2 * np.sin(q2)
    )

    return Y

def adaptive_tracking_control(
    q, dq,
    q_ref, dq_ref, ddq_ref,
    theta_hat,
    Ks=np.diag([10.0, 10.0]),
    K=np.diag([10.0, 10.0]),
    Gamma=np.diag([10.0, 10.0]),
):
    e = q_ref - q
    de = dq_ref - dq

    s = de + Ks @ e

    dqr = dq_ref + Ks @ e
    ddqr = ddq_ref + Ks @ de

    Y = regressor_matrix(q, dq, dqr, ddqr)

    tau = Y @ theta_hat + K @ s

    theta_hat_dot = Gamma @ Y.T @ s

    return tau, theta_hat_dot, s, Y



def trajectory_tracking_control(q, dq, q_ref, dq_ref, dqq_ref, model: DoublePendulum):
    """
    Compute the control torque tau for trajectory tracking control.
    q, dq: current joint positions and velocities
    q_ref, dq_ref: reference joint positions and velocities
    dqq_ref: reference joint accelerations
    model: DoublePendulum model for computing the dynamics
    """
    # Control gains
    Kp = np.diag([1.0, 1.0])  # Proportional gain
    Kd = np.diag([0.0, 0.0])     # Derivative gain

    # Compute the error terms
    e = q_ref - q          # Position error
    de = dq_ref - dq       # Velocity error

    # Compute the desired acceleration using PD control
    ddq_desired = Kp @ e + Kd @ de + dqq_ref   # Desired joint acceleration 

    # Compute the mass matrix, Coriolis matrix, and gravity vector
    M = model.mass_matrix(q)
    C = model.coriolis_matrix(q, dq)
    G = model.gravity_vector(q)

    # Compute the control torque using inverse dynamics
    tau = M @ ddq_desired + C @ dq + G

    return tau


def dynamics_state(x, tau, model: DoublePendulum):
    """
    State x = [q1, q2, dq1, dq2]
    Return dx = [dq1, dq2, ddq1, ddq2]
    """
    q = x[0:2]
    dq = x[2:4]

    ddq = model.forward_dynamics(q, dq, tau)

    dx = np.zeros_like(x)
    dx[0:2] = dq
    dx[2:4] = ddq

    return dx

def rk4_step(x, tau, dt, model: DoublePendulum):
    k1 = dynamics_state(x, tau, model)
    k2 = dynamics_state(x + 0.5 * dt * k1, tau, model)
    k3 = dynamics_state(x + 0.5 * dt * k2, tau, model)
    k4 = dynamics_state(x + dt * k3, tau, model)

    x_next = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x_next


def main():
    plant = DoublePendulum(
        m1=4.0,
        m2=2.0,
        l1=1.0,
        l2=1.0
    )

    q0 = np.array([0.0, 0.0])
    dq0 = np.array([0.0, 0.0])

    x = np.zeros(4)
    x[0:2] = q0
    x[2:4] = dq0

    theta_hat = np.array([1.0, 1.0])

    DT = 0.001
    T_move = 20.0
    steps = int(T_move / DT)

    t = 0.0

    q_log = []
    q_ref_log = []
    theta_log = []
    s_log = []
    time_log = []

    for i in range(steps):
        q = x[0:2]
        dq = x[2:4]

        q_ref, dq_ref, ddq_ref = joint_trajectory_rich(t)

        tau, theta_hat_dot, s, Y = adaptive_tracking_control(
            q, dq,
            q_ref, dq_ref, ddq_ref,
            theta_hat
        )

        x = rk4_step(x, tau, DT, plant)

        theta_hat = theta_hat + DT * theta_hat_dot

        q_log.append(q.copy())
        q_ref_log.append(q_ref.copy())
        theta_log.append(theta_hat.copy())
        s_log.append(s.copy())
        time_log.append(t)

        if i % 500 == 0:
            print(
                f"t = {t:.3f}, "
                f"q = {q}, "
                f"q_ref = {q_ref}, "
                f"theta_hat = {theta_hat}, "
                f"s = {s}"
            )

        t += DT

    q_log = np.array(q_log)
    q_ref_log = np.array(q_ref_log)
    theta_log = np.array(theta_log)
    s_log = np.array(s_log)
    time_log = np.array(time_log)

    plot_joint_trajectory_comparison_subplots(
        time_log,
        q_log,
        q_ref_log,
        ref_label="q_ref_log",
        est_label="q_log")
    plot_joint_trajectories(
        time_log,
        theta_log,
        labels=["m1_hat", "m2_hat"],
    )

    return time_log, q_log, q_ref_log, theta_log, s_log


if __name__ == "__main__":
    # file_path = os.path.abspath(".")
    # model_dir = os.path.join(os.path.abspath("."), "model")
    # print(model_dir)
    main()
