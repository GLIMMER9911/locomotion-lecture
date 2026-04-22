"""Control package exports.

Use lazy imports to avoid importing optional heavy GUI dependencies
(`pyqtgraph`/Qt) when only core simulation modules are needed.
"""

from importlib import import_module

__all__ = [
    "Kinematics",
    "MuJoCoSim",
    "KeyListener",
    "LowPassOnlineFilter",
    "MultiChartRealTimePlotManager",
    "ImpedanceController",
    "desired_trajectory",
    "DoublePendulum",
]

_LAZY_EXPORTS = {
    "Kinematics": ("src.pinocchio_kinematic", "Kinematics"),
    "MuJoCoSim": ("src.mujoco_viewer", "MuJoCoSim"),
    "KeyListener": ("src.key_listener", "KeyListener"),
    "LowPassOnlineFilter": ("src.lowpass_filter", "LowPassOnlineFilter"),
    "MultiChartRealTimePlotManager": ("src.matplot", "MultiChartRealTimePlotManager"),
    "ImpedanceController": ("src.impedance_controller", "ImpedanceController"),
    "desired_trajectory": ("src.impedance_controller", "desired_trajectory"),
    "DoublePendulum": ("src.double_pendulum", "DoublePendulum"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
