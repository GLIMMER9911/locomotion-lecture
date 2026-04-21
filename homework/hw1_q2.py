import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------
K = np.diag([2.0, 3.0, 4.0])


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,  x,   0.0]
    ])


def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    S = skew(axis)
    I = np.eye(3)
    return I + np.sin(theta) * S + (1.0 - np.cos(theta)) * (S @ S)


# ------------------------------------------------------------
# Euler XYZ convention: R = Rx(roll) Ry(pitch) Rz(yaw)
# Extraction formulas used here:
# pitch = asin(R[0,2])
# roll  = atan2(-R[1,2], R[2,2])
# yaw   = atan2(-R[0,1], R[0,0])
# ------------------------------------------------------------
def euler_xyz_from_R(R: np.ndarray) -> np.ndarray:
    # Numerical safeguard
    r02 = np.clip(R[0, 2], -1.0, 1.0)
    pitch = np.arcsin(r02)

    cp = np.cos(pitch)
    # Handle near-singular cases robustly
    if abs(cp) < 1e-8:
        # fallback
        roll = 0.0
        yaw = np.arctan2(R[1, 0], R[1, 1])
    else:
        roll = np.arctan2(-R[1, 2], R[2, 2])
        yaw = np.arctan2(-R[0, 1], R[0, 0])

    return np.array([roll, pitch, yaw])


# ------------------------------------------------------------
# Quaternion stiffness
# q = [eta, eps]
# eta = cos(theta/2)
# eps = axis * sin(theta/2)
# E(eta, eps) = eta I - S(eps)
# tau_Q = -2 E(eta, eps)^T K eps
# V_Q = 2 eps^T K eps
# ------------------------------------------------------------
def quaternion_torque_energy(axis: np.ndarray, theta: float):
    axis = axis / np.linalg.norm(axis)
    eta = np.cos(theta / 2.0)
    eps = axis * np.sin(theta / 2.0)
    E = eta * np.eye(3) - skew(eps)

    tau_q = -2.0 * E.T @ K @ eps
    V_q = 2.0 * eps.T @ K @ eps
    return tau_q, V_q


# ------------------------------------------------------------
# Euler stiffness
# tau_E = -K phi
# V_E = 1/2 phi^T K phi
# ------------------------------------------------------------
def euler_torque_energy(axis: np.ndarray, theta: float):
    R = rodrigues(axis, theta)
    phi = euler_xyz_from_R(R)
    tau_e = -K @ phi
    V_e = 0.5 * phi.T @ K @ phi
    return phi, tau_e, V_e


# ------------------------------------------------------------
# Sweep over theta for a given axis
# ------------------------------------------------------------
def evaluate_axis(axis: np.ndarray, thetas: np.ndarray):
    euler_angles = []
    tau_e_list = []
    V_e_list = []

    tau_q_list = []
    V_q_list = []

    for th in thetas:
        phi, tau_e, V_e = euler_torque_energy(axis, th)
        tau_q, V_q = quaternion_torque_energy(axis, th)

        euler_angles.append(phi)
        tau_e_list.append(tau_e)
        V_e_list.append(V_e)

        tau_q_list.append(tau_q)
        V_q_list.append(V_q)

    return {
        "phi": np.array(euler_angles),
        "tau_e": np.array(tau_e_list),
        "V_e": np.array(V_e_list),
        "tau_q": np.array(tau_q_list),
        "V_q": np.array(V_q_list),
    }


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_results(thetas: np.ndarray, result: dict, axis_name: str):
    theta_deg = thetas * 180.0 / np.pi

    tau_e = result["tau_e"]
    tau_q = result["tau_q"]

    tau_e_norm = np.linalg.norm(tau_e, axis=1)
    tau_q_norm = np.linalg.norm(tau_q, axis=1)

    V_e = result["V_e"]
    V_q = result["V_q"]    

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Torque and Energy Comparison ({axis_name})", fontsize=14)

    # --- τx ---
    ax = axes[0, 0]
    ax.plot(theta_deg, tau_e[:, 0], label="Euler")
    ax.plot(theta_deg, tau_q[:, 0], label="Quaternion")
    ax.set_title(r"$\tau_x$")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel("Torque")
    ax.grid(True)

    # --- τy ---
    ax = axes[0, 1]
    ax.plot(theta_deg, tau_e[:, 1], label="Euler")
    ax.plot(theta_deg, tau_q[:, 1], label="Quaternion")
    ax.set_title(r"$\tau_y$")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel("Torque")
    ax.grid(True)

    # --- τz ---
    ax = axes[1, 0]
    ax.plot(theta_deg, tau_e[:, 2], label="Euler")
    ax.plot(theta_deg, tau_q[:, 2], label="Quaternion")
    ax.set_title(r"$\tau_z$")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel("Torque")
    ax.grid(True)

    # --- ‖τ‖ ---
    ax = axes[1, 1]
    ax.plot(theta_deg, tau_e_norm, label="Euler")
    ax.plot(theta_deg, tau_q_norm, label="Quaternion")
    ax.set_title(r"$\|\tau\|$")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel("Norm")
    ax.grid(True)

    # --- Energy ---
    ax = axes[2, 0]
    ax.plot(theta_deg, V_e, label="Euler")
    ax.plot(theta_deg, V_q, label="Quaternion")
    ax.set_title("Energy")
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel("Energy")
    ax.grid(True)

    # --- 最后一个子图放 legend ---
    ax = axes[2, 1]
    ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="center", fontsize=12)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    thetas = np.linspace(-np.pi, np.pi, 1000)

    axis1 = np.array([1.0, 0.0, 0.0])
    axis2 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

    result1 = evaluate_axis(axis1, thetas)
    result2 = evaluate_axis(axis2, thetas)

    plot_results(thetas, result1, r"$e_\omega = [1,0,0]^T$")
    plot_results(thetas, result2, r"$e_\omega = \frac{1}{\sqrt{2}}[1,1,0]^T$")

    # Print a few symbolic checkpoints
    print("Axis 1, theta = pi/4")
    th = np.pi / 4
    phi1, tau_e1, V_e1 = euler_torque_energy(axis1, th)
    tau_q1, V_q1 = quaternion_torque_energy(axis1, th)
    print("Euler angles:", phi1)
    print("Euler torque:", tau_e1)
    print("Quaternion torque:", tau_q1)
    print("Euler energy:", V_e1)
    print("Quaternion energy:", V_q1)
    print()

    print("Axis 2, theta = pi/4")
    phi2, tau_e2, V_e2 = euler_torque_energy(axis2, th)
    tau_q2, V_q2 = quaternion_torque_energy(axis2, th)
    print("Euler angles:", phi2)
    print("Euler torque:", tau_e2)
    print("Quaternion torque:", tau_q2)
    print("Euler energy:", V_e2)
    print("Quaternion energy:", V_q2)

    plt.show()