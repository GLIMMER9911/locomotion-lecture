"""Joint-space impedance controller and reference trajectory."""

import numpy as np


def desired_trajectory(t: float, q0: np.ndarray, q_goal: np.ndarray, T_move: float):
    """
    Linear interpolation from q0 to q_goal over [0, T_move]; constant after T_move.

    Returns:
        q_des, dq_des, ddq_des
    """
    if t >= T_move:
        return q_goal.copy(), np.zeros_like(q_goal), np.zeros_like(q_goal)
    s = t / T_move
    q_des =  (1 - s) * q0 + s * q_goal
    dq_des = (q_goal - q0) / T_move
    ddq_des = np.zeros_like(q0)
    return q_des, dq_des, ddq_des


class ImpedanceController:
    """
    Joint-space impedance control: M(q) * (ddq_des + Kd*de + Kp*e) + nle.
    """

    def __init__(self, nq: int, 
        Kp: np.ndarray = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]), 
        Kd: np.ndarray = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
        ):
        self.nq = nq
        if Kp.shape[0] != nq:
            raise ValueError(f"Kp must have {nq} elements")
        if Kd.shape[0] != nq:
            raise ValueError(f"Kd must have {nq} elements")
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)


    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
        M: np.ndarray,
        nle: np.ndarray,
    ) -> np.ndarray:
        """
        Compute control torque.

        Args:
            q, dq: Current joint position and velocity.
            q_des, dq_des, ddq_des: Desired trajectory.
            M, nle: Mass matrix and nonlinear effects from compute_pin_dynamics.
        """
        if ddq_des is None:
            ddq_des = np.zeros_like(q)
        e = q_des - q
        de = dq_des - dq
        tau = M @ (ddq_des + self.Kd @ de + self.Kp @ e) + nle
        return tau


class MomentumBasedDisturbanceObserver:
    def __init__(self, nq: int, gain):
        self.nq = nq

        gain = np.asarray(gain, dtype=float)
        if gain.ndim == 0:
            gain = np.full(nq, float(gain))
        if gain.shape != (nq,):
            raise ValueError(f"gain must have shape {(nq,)}, got {gain.shape}")

        # diagonal gain: K = diag(gain)
        self.gain = gain

        # estimated momentum
        self.p_hat = np.zeros(nq)

        # estimated disturbance torque
        self.tau_hat = np.zeros(nq)

        self.initialized = False

    def reset(self):
        self.p_hat[:] = 0.0
        self.tau_hat[:] = 0.0
        self.initialized = False

    def update(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        M: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
        tau_applied: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        dq = np.asarray(dq, dtype=float)
        M = np.asarray(M, dtype=float)
        C = np.asarray(C, dtype=float)
        g = np.asarray(g, dtype=float)
        tau_applied = np.asarray(tau_applied, dtype=float)

        if dq.shape != (self.nq,):
            raise ValueError(f"dq must have shape {(self.nq,)}, got {dq.shape}")
        if M.shape != (self.nq, self.nq):
            raise ValueError(f"M must have shape {(self.nq, self.nq)}, got {M.shape}")
        if C.shape != (self.nq, self.nq):
            raise ValueError(f"C must have shape {(self.nq, self.nq)}, got {C.shape}")
        if g.shape != (self.nq,):
            raise ValueError(f"g must have shape {(self.nq,)}, got {g.shape}")
        if tau_applied.shape != (self.nq,):
            raise ValueError(
                f"tau_applied must have shape {(self.nq,)}, got {tau_applied.shape}"
            )

        # true generalized momentum
        p = M @ dq

        # optional: initialize p_hat to p for a cleaner startup
        if not self.initialized:
            self.p_hat = p.copy()
            self.tau_hat[:] = 0.0
            self.initialized = True
            return self.tau_hat.copy()
        
        self.tau_hat = self.gain * (p - self.p_hat)

        # p_hat_dot = C^T dq - g + tau + tau_hat
        p_hat_dot = C.T @ dq - g + tau_applied + self.tau_hat

        # Euler integration
        self.p_hat = self.p_hat + dt * p_hat_dot
        
        return self.tau_hat.copy()
