"""MuJoCo simulation interface: load model, viewer, state sync, and stepping."""

import warnings

# Suppress GLFW "not initialized" warning during viewer/process teardown (benign).
warnings.filterwarnings("ignore", message=".*GLFW.*not initialized.*")

import numpy as np
import mujoco
import mujoco.viewer


class MuJoCoSim:
    """Wrapper for MuJoCo model, data, viewer, and joint index mapping."""

    def __init__(self, scene_path: str, nq: int):
        """
        Load MuJoCo scene and allocate state.

        Args:
            scene_path: Path to scene XML (e.g. model/scene.xml).
            nq: Number of position DOFs (must match joint indices used for control).
        """
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.nq = nq
        self.joint_indices = np.arange(0, nq)
        self.control_indices = self._infer_control_indices()
        self._viewer = None

        mujoco.mj_forward(self.model, self.data)

    def _infer_control_indices(self) -> np.ndarray:
        """Pick the actuator slots that should receive nq-dimensional torque commands."""
        if self.model.nu == self.nq:
            return np.arange(self.model.nu)

        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(self.model.nu)
        ]
        torque_indices = [i for i, name in enumerate(actuator_names) if name.endswith("_tau")]
        if len(torque_indices) == self.nq:
            return np.asarray(torque_indices, dtype=int)

        raise ValueError(
            f"Unable to map {self.nq} torque controls onto model.nu={self.model.nu} actuators. "
            f"Actuators found: {actuator_names}"
        )

    def get_model_np(self):
        return self.model.nq, self.model.nv, self.model.nu

    def is_running(self):
        return self._viewer.is_running()

    def launch_viewer(self):
        """Start passive viewer. Call sync() after each step to update."""
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def forward_passive_viewer(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.viewer.launch_passive(self.model, self.data)
        self.sync_viewer()

    def get_joint_state(self):
        """Return (q, dq) for the controlled joints from current MuJoCo state."""
        q = self.data.qpos[self.joint_indices].copy()
        dq = self.data.qvel[self.joint_indices].copy()
        return q, dq

    def set_joint_positions(self, q: np.ndarray):
        """Set qpos for controlled joints and run mj_forward."""
        self.data.qpos[self.joint_indices] = q
        mujoco.mj_forward(self.model, self.data)

    def get_joint_torque(self):
        """Return joint torques for the controlled joints."""
        mujoco.mj_inverse(self.model, self.data)

        tau_cmd = self.data.qfrc_actuator.copy()
        tau_total = self.data.qfrc_inverse.copy()
        return tau_cmd.copy(), tau_total.copy()


    def set_control(self, tau: np.ndarray):
        """Set control (joint torques) for the next step."""
        tau = np.asarray(tau)
        if tau.shape != (self.nq,):
            raise ValueError(f"Expected tau shape {(self.nq,)}, got {tau.shape}")

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.control_indices] = tau

    def get_site_id(self, site_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def get_site_jacobian(self, site_id: int):
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
        return jac_pos[:, self.joint_indices], jac_rot[:, self.joint_indices]
    
    def set_external_force(self, site_id: int, force: np.ndarray):
        self.data.qfrc_applied[:] = 0.0
        point = self.data.site_xpos[site_id]
        assert force.shape == (6,), f"Force must be a 6D vector: {force.shape}"
        ext_force  = force[:3]
        ext_torque = force[3:]
        body_id = self.model.site_bodyid[site_id]
        mujoco.mj_applyFT(self.model, self.data, 
                        ext_force, ext_torque, point, body_id, self.data.qfrc_applied)
        return self.data.qfrc_applied.copy()
    
    def step(self):
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def sync_viewer(self):
        """Sync passive viewer with current state."""
        if self._viewer is not None:
            self._viewer.sync()

    def close_viewer(self):
        """Close the passive viewer and release GLFW/resources."""
        if self._viewer is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress GLFW teardown warnings
                    self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def run_loop(self):
        while self.is_running():
            self.step()
            self.sync_viewer()

    @property
    def dt(self) -> float:
        """Simulation timestep."""
        return self.model.opt.timestep
