"""
double pendulum trajectory tracking control with MuJoCo.
Dynamics is given 
Run :
    python elastic_motor_control.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# Desired trajectory: step input
# ============================================================

def desired_trajectory(t, td=0.2):
    """
    Step input:
        q_d = 0,   t < td
        q_d = 0.5, t >= td

    For this homework, qd_dot and qd_ddot are set to zero.
    """
    if t < td:
        qd = 0.0
    else:
        qd = 0.5

    qd_dot = 0.0
    qd_ddot = 0.0

    return qd, qd_dot, qd_ddot


# ============================================================
# Cascaded controller
# state: x = [q, q_dot, theta, theta_dot]
# system:
#   m q_ddot + C q_dot + k(q-theta) + G(q) = 0
#   b theta_ddot + k(theta - q) = tau_m
# Define elastic torque:
#   tau = k(theta - q)  
# Outer PD+:
#   tau_d = m (qdd_d + Kd (qd_dot - q_dot) + Kp (qd - q))
# Inner torque controller:
#   Make tau track tau_d by commanding theta_ddot.
# ============================================================

def cascaded_controller(t, x, params, gains):
    q, q_dot, theta, theta_dot = x

    m = params["m"]
    b = params["b"]
    k = params["k"]
    d_q = params.get("d_q", 0.0)
    d_theta = params.get("d_theta", 0.0)

    Kp_q = gains["Kp_q"]
    Kd_q = gains["Kd_q"]

    Kp_tau = gains["Kp_tau"]
    Kd_tau = gains["Kd_tau"]

    qd, qd_dot, qd_ddot = desired_trajectory(t)

    # Elastic torque
    tau = k * (theta - q)

    # Current link acceleration
    q_ddot = (tau - d_q * q_dot) / m

    # Elastic torque derivative
    tau_dot = k * (theta_dot - q_dot)

    # Outer PD+ desired joint torque
    tau_d = m * (
        qd_ddot
        + Kd_q * (qd_dot - q_dot)
        + Kp_q * (qd - q)
    )

    # Approximate derivative of desired torque
    # Since qd_dot = qd_ddot = 0 for a step after td:
    tau_d_dot = m * (
        -Kp_q * q_dot
        -Kd_q * q_ddot
    )

    # Inner torque tracking error
    e_tau = tau_d - tau
    e_tau_dot = tau_d_dot - tau_dot

    # Desired elastic torque acceleration
    tau_ddot_cmd = Kp_tau * e_tau + Kd_tau * e_tau_dot

    # Because:
    # tau = k(theta - q)
    # tau_ddot = k(theta_ddot - q_ddot)
    #
    # Therefore:
    # theta_ddot_cmd = q_ddot + tau_ddot_cmd / k
    theta_ddot_cmd = q_ddot + tau_ddot_cmd / k

    # Motor torque command:
    # b theta_ddot + tau + d_theta theta_dot = tau_m
    tau_m = b * theta_ddot_cmd + tau + d_theta * theta_dot

    return tau_m, tau, tau_d, qd


# ============================================================
# Dynamics
# ============================================================

def elastic_joint_dynamics(t, x, params, gains):
    """
    x = [q, q_dot, theta, theta_dot]
    """

    q, q_dot, theta, theta_dot = x

    m = params["m"]
    b = params["b"]
    k = params["k"]
    d_q = params.get("d_q", 0.0)
    d_theta = params.get("d_theta", 0.0)

    tau_m, tau, tau_d, qd = cascaded_controller(t, x, params, gains)

    q_ddot = (tau - d_q * q_dot) / m

    theta_ddot = (tau_m - tau - d_theta * theta_dot) / b

    dx = np.array([
        q_dot,
        q_ddot,
        theta_dot,
        theta_ddot
    ])

    return dx


# ============================================================
# Simulation function
# ============================================================

def simulate_case(case_name, params, gains, t_final=2.0):
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    t_eval = np.linspace(0.0, t_final, 3000)

    sol = solve_ivp(
        fun=lambda t, x: elastic_joint_dynamics(t, x, params, gains),
        t_span=(0.0, t_final),
        y0=x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )

    t = sol.t
    q = sol.y[0]
    q_dot = sol.y[1]
    theta = sol.y[2]
    theta_dot = sol.y[3]

    tau_m_log = []
    tau_log = []
    tau_d_log = []
    qd_log = []

    for i in range(len(t)):
        x_i = sol.y[:, i]
        tau_m, tau, tau_d, qd = cascaded_controller(t[i], x_i, params, gains)

        tau_m_log.append(tau_m)
        tau_log.append(tau)
        tau_d_log.append(tau_d)
        qd_log.append(qd)

    result = {
        "case_name": case_name,
        "t": t,
        "q": q,
        "q_dot": q_dot,
        "theta": theta,
        "theta_dot": theta_dot,
        "tau_m": np.array(tau_m_log),
        "tau": np.array(tau_log),
        "tau_d": np.array(tau_d_log),
        "qd": np.array(qd_log),
        "params": params
    }

    return result


# ============================================================
# Plot function
# ============================================================

def plot_result(result):
    t = result["t"]
    q = result["q"]
    theta = result["theta"]
    qd = result["qd"]
    tau_m = result["tau_m"]

    case_name = result["case_name"]

    plt.figure(figsize=(8, 5))
    plt.plot(t, qd, "k--", linewidth=2, label=r"$q_d$")
    plt.plot(t, q, linewidth=2, label=r"$q$")
    plt.plot(t, theta, linewidth=2, label=r"$\theta$")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.title(f"Position response: {case_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(t, tau_m, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel(r"Motor torque $\tau_m$ [Nm]")
    plt.title(f"Motor torque input: {case_name}")
    plt.grid(True)
    plt.tight_layout()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Nominal system parameters
    # --------------------------------------------------------
    nominal_params = {
        "m": 2.0,
        "b": 0.75,
        "k": 300.0,
        "d_q": 0.0,
        "d_theta": 0.0
    }

    omega_q = 8.0
    omega_tau = 60.0

    gains = {
        "Kp_q": omega_q ** 2,
        "Kd_q": 2.0 * omega_q,

        "Kp_tau": omega_tau ** 2,
        "Kd_tau": 2.0 * omega_tau
    }

    # --------------------------------------------------------
    # Case 1: nominal
    # --------------------------------------------------------
    case_nominal = simulate_case(
        "Nominal: m=2, b=0.75, k=300",
        nominal_params,
        gains
    )

    # --------------------------------------------------------
    # Case 2: reduced stiffness
    # --------------------------------------------------------
    reduced_stiffness_params = {
        "m": 2.0,
        "b": 0.75,
        "k": 10.0,
        "d_q": 0.0,
        "d_theta": 0.0
    }

    case_low_k = simulate_case(
        "Reduced stiffness: k=10",
        reduced_stiffness_params,
        gains
    )

    # --------------------------------------------------------
    # Case 3: increased motor inertia
    # --------------------------------------------------------
    increased_motor_inertia_params = {
        "m": 2.0,
        "b": 15.0,
        "k": 300.0,
        "d_q": 0.0,
        "d_theta": 0.0
    }

    case_high_b = simulate_case(
        "Increased motor inertia: b=15",
        increased_motor_inertia_params,
        gains
    )

    # --------------------------------------------------------
    # Case 4: viscous friction at the joint
    # --------------------------------------------------------
    joint_friction_params = {
        "m": 2.0,
        "b": 0.75,
        "k": 300.0,
        "d_q": 3.0,
        "d_theta": 0.0
    }

    case_joint_friction = simulate_case(
        "Joint-side viscous friction: d_q=3",
        joint_friction_params,
        gains
    )

    # --------------------------------------------------------
    # Case 5: viscous friction at the motor
    # --------------------------------------------------------
    motor_friction_params = {
        "m": 2.0,
        "b": 0.75,
        "k": 300.0,
        "d_q": 0.0,
        "d_theta": 3.0
    }

    case_motor_friction = simulate_case(
        "Motor-side viscous friction: d_theta=3",
        motor_friction_params,
        gains
    )

    # --------------------------------------------------------
    # Plot all results
    # --------------------------------------------------------
    results = [
        case_nominal,
        case_low_k,
        case_high_b,
        case_joint_friction,
        case_motor_friction
    ]

    for res in results:
        plot_result(res)

    plt.show()