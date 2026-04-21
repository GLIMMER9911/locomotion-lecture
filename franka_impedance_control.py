"""
franka inverse dynamics control with MuJoCo + Pinocchio.
Run :
    python franka_impedance_control.py
"""
import os
import sys
import numpy as np
import pinocchio as pin
from pathlib import Path
from src.mujoco_viewer import MuJoCoSim
from src.impedance_controller import (
    ImpedanceController,
    MomentumBasedDisturbanceObserver,
    desired_trajectory,
)
from src.matplot import MultiChartRealTimePlotManager
from src.plotting import plot_joint_trajectories, plot_joint_trajectory_comparison_subplots


def compute_pin_dynamics(model, data, q:np.ndarray, dq:np.ndarray):
    M = pin.crba(model, data, q)
    M = 0.5 * (M + M.T) # ensure symmetry
    C = pin.computeCoriolisMatrix(model, data, q, dq)
    nle = pin.nonLinearEffects(model, data, q, dq)
    g = pin.computeGeneralizedGravity(model, data, q)
    return M, C, nle, g

def main(
    model_dir: str = None,
    urdf_dir: str = "franka_panda_urdf/robots/panda_arm.urdf",
    scene_dir: str = "franka_emika_panda/scene.xml",
):
    if model_dir is None:
        model_dir = os.path.join(os.path.abspath("."), "model")
    urdf_dir = os.path.join(model_dir, urdf_dir)
    scene_dir = os.path.join(model_dir, scene_dir)

    # Pinocchio model
    model = pin.buildModelFromUrdf(urdf_dir)
    data = model.createData()
    nq = model.nq
    print(f"Number of joints: {nq}")

    # MuJoCo model
    sim = MuJoCoSim(scene_dir, nq)
    # Joint initial positions
    joint_initial_pos = np.array([
        0.0, -0.7854, 0.0, -2.35621, 0.0, 1.5708, 0.0,
    ])

    sim.set_joint_positions(joint_initial_pos)
    sim.launch_viewer()
    sim.sync_viewer()

    q0, _ = sim.get_joint_state()
    print(f"Initial joint positions: {q0}")

    # Desired joint positions
    desired_joint_pos = joint_initial_pos + np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])
    # desired_joint_pos = joint_initial_pos + 0.5
    print(f"Desired joint positions: {desired_joint_pos}")

    DT = sim.dt
    print(f"DT: {DT}")

    # controller initialization
    Kp = np.array([25.0, 25.0, 25.0, 25.0, 50.0, 25.0, 200.0])
    Kd = np.array([10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 100.0])
    controller = ImpedanceController(nq, Kp, Kd)
    observer = MomentumBasedDisturbanceObserver(
        nq,
        gain=np.array([200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]),
    )

    ## external force on the end effector
    site_id = sim.get_site_id("attachment_site")
    if site_id == -1:
        raise ValueError("Site ID not found")
        sys.exit(1)
    print(f"Site ID: {site_id}")
    external_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # [fx, fy, fz, tx, ty, tz]

    # <!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

    # print(f"DT: {DT}")
    T_move = 3.0
    step = T_move/DT + 2000
    external_force_time_start = 1.0
    external_force_time_end = 2.0

    # log data
    show_plot = False
    log_t = []
    log_q = []
    log_dq = []
    log_q_des = []
    log_dq_des = []
    log_tau_hat = []
    log_tau_ext = []
    t = 0.0

    # 
    if show_plot:
        plot_manager = MultiChartRealTimePlotManager()
        plot_manager.addNewFigurePlotter("q", title="q", row=0, col=0)
        plot_manager.addNewFigurePlotter("dq", title="dq", row=0, col=1)
        plot_manager.addNewFigurePlotter("q_des", title="q_des", row=1, col=0)
        plot_manager.addNewFigurePlotter("dq_des", title="dq_des", row=1, col=1)

        colors = ["r", "g", "b", "c", "m", "y", "w"]
        for j in range(nq):
            plot_manager.addPlotToPlotter("q", f"q{j}", color=colors[j % len(colors)])
            plot_manager.addPlotToPlotter("dq", f"dq{j}", color=colors[j % len(colors)])
            plot_manager.addPlotToPlotter("q_des", f"q_des{j}", color=colors[j % len(colors)])
            plot_manager.addPlotToPlotter("dq_des", f"dq_des{j}", color=colors[j % len(colors)])

    try:
        for _ in range(int(step)):
            if t > external_force_time_start and t < external_force_time_end:
                external_force[0] = 6.0
                external_force[1] = 8.0
                external_force[2] = 10.0

                # print("External force applied")
            else:
                external_force[0] = 0.0
                external_force[1] = 0.0
                external_force[2] = 0.0

            q, dq = sim.get_joint_state()
            q_des, dq_des, ddq_des = desired_trajectory(t, q0, desired_joint_pos, T_move)

            M, C, nle, g = compute_pin_dynamics(model, data, q, dq)
            tau = controller.compute_torque(q, dq, q_des, dq_des, ddq_des, M, nle)
            tau_hat = observer.update(q, dq, M, C, g, tau, DT)


            jac_pos, jac_rot = sim.get_site_jacobian(site_id)
            spatial_jacobian = np.vstack((jac_pos, jac_rot))
            # Generalized disturbance torque induced by the Cartesian wrench.
            # This is the quantity the momentum observer should converge to.
            tau_ext = spatial_jacobian.T @ external_force

            sim.set_control(tau)
            sim.set_external_force(site_id, external_force)

            sim.step()
            sim.sync_viewer()

            if show_plot:
                for j in range(nq):
                    plot_manager.updateDataToPlotter("q", f"q{j}", float(q[j]))
                    plot_manager.updateDataToPlotter("dq", f"dq{j}", float(dq[j]))
                    plot_manager.updateDataToPlotter("q_des", f"q_des{j}", float(q_des[j]))
                    plot_manager.updateDataToPlotter("dq_des", f"dq_des{j}", float(dq_des[j]))
            
            log_t.append(t)
            log_q.append(q.copy())
            log_dq.append(dq.copy())
            log_q_des.append(q_des.copy())
            log_dq_des.append(dq_des.copy())
            log_tau_hat.append(tau_hat.copy())
            log_tau_ext.append(tau_ext.copy())

            if int(t / DT) % 500 == 0:
                print(
                    f"t={t:5.3f} | ||tau_nom||={np.linalg.norm(tau):.3f} | "
                    f"||tau_hat||={np.linalg.norm(tau_hat):.3f} | "
                    f"||tau_ext||={np.linalg.norm(tau_ext):.3f} | "
                    f"force={external_force[:3]}"
                )

            t += DT

        plot_joint_trajectory_comparison_subplots(
            log_t,
            log_tau_ext,
            log_tau_hat,
            ref_label="tau_e",
            est_label="tau_hat_e",
        )

        plot_joint_trajectory_comparison_subplots(
            log_t,
            log_q_des,
            log_q,
            ref_label="q_des",
            est_label="q",
        )
        plot_joint_trajectory_comparison_subplots(
            log_t,
            log_dq_des,
            log_dq,
            ref_label="dq_des",
            est_label="dq"
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        if show_plot:
            plot_manager.closeAll()
        sim.close_viewer()

if __name__ == "__main__":
    file_path = os.path.abspath(".")
    model_dir = os.path.join(os.path.abspath("."), "model")
    # print(model_dir)
    main(model_dir=model_dir)
