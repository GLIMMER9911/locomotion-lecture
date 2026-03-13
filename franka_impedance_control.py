"""
franka inverse dynamics control with MuJoCo + Pinocchio.
Run :
    python franka_impedance_control.py
"""
import os
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi
from pathlib import Path
from src.mujoco_viewer import MuJoCoSim
from src.impedance_controller import ImpedanceController, desired_trajectory
from src.matplot import MultiChartRealTimePlotManager


def compute_pin_dynamics(model, data, q:np.ndarray, dq:np.ndarray):
    M = pin.crba(model, data, q)
    M = 0.5 * (M + M.T) # ensure symmetry
    nle = pin.nonLinearEffects(model, data, q, dq)
    return M, nle

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
    q0, dq0 = sim.get_joint_state()

    # Desired joint positions
    desired_joint_pos = joint_initial_pos + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])

    # controller initialization
    Kp = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    Kd = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0])
    controller = ImpedanceController(nq, Kp, Kd)

    DT = sim.dt
    sim_time = 10.0
    steps = int(sim_time / DT)

    # log data
    log_t = []
    log_q = []
    log_dq = []
    t = 0.0

    # 
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

    for i in range(steps):
        q, dq = sim.get_joint_state()
        q_des, dq_des, ddq_des = desired_trajectory(t, q0, desired_joint_pos, sim_time)
        M, nle = compute_pin_dynamics(model, data, q, dq)
        tau = controller.compute_torque(q, dq, q_des, dq_des, ddq_des, M, nle)
        sim.set_control(tau)
        sim.step()
        sim.sync_viewer()

        for j in range(nq):
            plot_manager.updateDataToPlotter("q", f"q{j}", float(q[j]))
            plot_manager.updateDataToPlotter("dq", f"dq{j}", float(dq[j]))
            plot_manager.updateDataToPlotter("q_des", f"q_des{j}", float(q_des[j]))
            plot_manager.updateDataToPlotter("dq_des", f"dq_des{j}", float(dq_des[j]))

        # log data
        t += DT
        log_t.append(t)
        log_q.append(q.copy())
        log_dq.append(dq.copy())

    sim.close_viewer()

if __name__ == "__main__":
    file_path = os.path.abspath(".")
    model_dir = os.path.join(os.path.abspath("."), "model")
    # print(model_dir)
    main(model_dir=model_dir)