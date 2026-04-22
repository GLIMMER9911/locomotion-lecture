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

def main(
    model_dir: str = None,
    scene_dir: str = "double_pendulum/doublependulum.xml",
):
    if model_dir is None:
        model_dir = os.path.join(os.path.abspath("."), "model")
    scene_dir = os.path.join(model_dir, scene_dir)

    # DoublePendulum model
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    model = DoublePendulum(m1, m2, l1, l2)


    # MuJoCo model
    nq = 2   
    sim = MuJoCoSim(scene_dir, nq)

    # Joint initial positions
    joint_initial_pos = np.array([
        0.0, 0.0,
    ])

    sim.set_joint_positions(joint_initial_pos)
    sim.launch_viewer()
    sim.sync_viewer()

    q0, _ = sim.get_joint_state()
    print(f"Initial joint positions: {q0}")

    DT = sim.dt
    print(f"DT: {DT}")

    T_move = 3.0
    step = T_move/DT + 2000

    t = 0.0

    while True:
        for _ in range(int(step)):

            q, dq = sim.get_joint_state()

            tau = np.random.uniform(-1, 1, size=(nq,)) # random torque for testing

            sim.set_control(tau)

            sim.step()
            sim.sync_viewer()
            


            if int(t / DT) % 500 == 0:
                # print(
                #     f"t={t:5.3f} | ||tau_nom||={np.linalg.norm(tau):.3f} | "
                #     f"||tau_hat||={np.linalg.norm(tau_hat):.3f} | "
                #     f"||tau_ext||={np.linalg.norm(tau_ext):.3f} | "
                #     f"force={external_force[:3]}"
                # )
                pass

            t += DT

        # plot_joint_trajectory_comparison_subplots(
        #     log_t,
        #     log_tau_ext,
        #     log_tau_hat,
        #     ref_label="tau_e",
        #     est_label="tau_hat_e",
        # )

    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt")
    # finally:
    #     pass
    #     # sim.close_viewer()

if __name__ == "__main__":
    file_path = os.path.abspath(".")
    model_dir = os.path.join(os.path.abspath("."), "model")
    # print(model_dir)
    main(model_dir=model_dir)
