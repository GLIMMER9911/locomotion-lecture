"""double pendulum dynamics."""

import numpy as np

class DoublePendulum:

    def __init__(self, 
                 m1 = 1.0, 
                 m2 = 1.0, 
                 l1 = 1.0, 
                 l2 = 1.0):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = 9.81

    def mass_matrix(self, q:np.ndarray):
        """ M(q), q = [q1, q2]"""
        q1, q2 = q
        
        c21 = np.cos(q2 - q1)

        M = np.array([
            [(self.m1 + self.m2) * self.l1**2, self.m2 * self.l1* self.l2 * c21],
            [self.m2 * self.l1 * self.l2 * c21, self.m2 * self.l1 * self.l2 * c21],
        ], dtype = float)

        return M
    
    def coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        C(q, dq)
        so that C(q, dq) @ dq matches the term in the assignment.
        """
        q1, q2 = q
        dq1, dq2 = dq
        m2, l1, l2 = self.m2, self.l1, self.l2

        s21 = np.sin(q2 - q1)

        C = np.array([
            [0.0,                   -m2 * l1 * l2 * s21 * dq2],
            [m2 * l1 * l2 * s21 * dq1,   0.0]
        ], dtype=float)

        return C
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        g(q)
        """
        q1, q2 = q
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        G = np.array([
            (m1 + m2) * g * l1 * np.sin(q1),
            m2 * g * l2 * np.sin(q2)
        ], dtype=float)

        return G
    
    def inv_dynamics(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray
    ) -> np.ndarray:
        """
        Inverse dynamics:
            tau = M(q) ddq + C(q, dq) dq + g(q)
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        G = self.gravity_vector(q)

        tau = M @ ddq + C @ dq + G
        return tau

    def forward_dynamics(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        Forward dynamics:
            ddq = M(q)^(-1) [tau - C(q,dq)dq - g(q)]
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, dq)
        G = self.gravity_vector(q)

        rhs = tau - C @ dq - G
        ddq = np.linalg.solve(M, rhs)
        return ddq
        
if __name__ == "__main__":
    robot = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)

    q = np.array([0.1, 0.2])
    dq = np.array([0.0, 0.0])
    ddq = np.array([0.5, -0.3])

    tau = robot.inv_dynamics(q, dq, ddq)
    print("tau =", tau)

    ddq_check = robot.forward_dynamics(q, dq, tau)
    print("ddq_check =", ddq_check)