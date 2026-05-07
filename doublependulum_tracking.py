"""
double pendulum trajectory tracking control with MuJoCo.
Dynamics is given 
Run :
    python doublependulum_tracking.py
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



def trajectory_tracking_control(q, dq, q_ref, dq_ref, dqq_ref, model: DoublePendulum):
    """
    Compute the control torque tau for trajectory tracking control.
    q, dq: current joint positions and velocities
    q_ref, dq_ref: reference joint positions and velocities
    dqq_ref: reference joint accelerations
    model: DoublePendulum model for computing the dynamics
    """
    # Control gains
    Kp = np.diag([10.0, 10.0])  # Proportional gain
    Kd = np.diag([2.0, 2.0])     # Derivative gain

    # Compute the error terms
    e = q_ref - q          # Position error
    de = dq_ref - dq       # Velocity error

    # Compute the desired acceleration using PD control
    ddq_desired = Kp @ e + Kd @ de + dqq_ref  # Desired joint acceleration

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


def main(
):
    # Controller model
    m1_model = 1.0
    m2_model = 1.0
    l1 = 1.0
    l2 = 1.0
    model = DoublePendulum(m1_model, m2_model, l1, l2)

    # Initial state
    q0 = np.array([0.0, 0.0])
    dq0 = np.array([0.0, 0.0])

    x = np.zeros(4)
    x[0:2] = q0
    x[2:4] = dq0

    DT = 0.001
    T_move = 10.0
    steps = int(T_move / DT)

    t = 0.0

    q_log = []
    q_ref_log = []
    tau_log = []
    time_log = []

    for i in range(steps):
        q = x[0:2]
        dq = x[2:4]

        q_ref, dq_ref, ddq_ref = joint_trajectory(t)

        tau = trajectory_tracking_control(
            q, dq,
            q_ref, dq_ref, ddq_ref,
            model
        )

        x = rk4_step(x, tau, DT, model)

        q_log.append(q.copy())
        q_ref_log.append(q_ref.copy())
        tau_log.append(tau.copy())
        time_log.append(t)

        if i % 500 == 0:
            print(
                f"t = {t:.3f}, "
                f"q = {q}, "
                f"q_ref = {q_ref}, "
                f"error = {q_ref - q}"
            )

        t += DT

    q_log = np.array(q_log)
    q_ref_log = np.array(q_ref_log)
    tau_log = np.array(tau_log)
    time_log = np.array(time_log)

    plot_joint_trajectory_comparison_subplots(
    time_log,
    q_log,
    q_ref_log,
    ref_label="q_ref_log",
    est_label="q_log",
)

    return time_log, q_log, q_ref_log, tau_log



if __name__ == "__main__":
    # file_path = os.path.abspath(".")
    # model_dir = os.path.join(os.path.abspath("."), "model")
    # print(model_dir)
    main()
