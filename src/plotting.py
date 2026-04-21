"""Plotting utilities for simulation logs."""

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_trajectories(
    log_t: list,
    log_q,
    joint_indices: list = None,
    labels: list = None,
    figsize=(10, 5),
):
    """
    Plot selected joint positions over time.

    Args:
        log_t: List of time values.
        log_q: Sequence or array with shape (n_steps, nq) of joint positions.
        joint_indices: Which joint indices to plot (default all joints).
        labels: Legend labels (default "Joint 0", "Joint 1", etc.).
    """
    log_t = np.asarray(log_t)
    log_q = np.asarray(log_q)

    if log_q.ndim != 2:
        raise ValueError(
            f"log_q must have shape (n_steps, nq), got array with shape {log_q.shape}"
        )

    if log_t.shape[0] != log_q.shape[0]:
        raise ValueError(
            f"log_t and log_q must have the same length, got {log_t.shape[0]} and {log_q.shape[0]}"
        )

    if joint_indices is None:
        joint_indices = list(range(log_q.shape[1]))

    invalid_indices = [j for j in joint_indices if j < 0 or j >= log_q.shape[1]]
    if invalid_indices:
        raise IndexError(
            f"joint_indices {invalid_indices} out of bounds for {log_q.shape[1]} joints"
        )

    if labels is None:
        labels = [f"Joint {j}" for j in joint_indices]

    if len(labels) != len(joint_indices):
        raise ValueError("labels and joint_indices must have the same length")

    plt.figure(figsize=figsize)
    for j, label in zip(joint_indices, labels):
        plt.plot(log_t, log_q[:, j], label=label)
    plt.grid()
    plt.legend()
    plt.show()


def plot_joint_trajectory_comparison(
    log_t: list,
    series_ref,
    series_est,
    joint_indices: list = None,
    ref_prefix: str = "ref",
    est_prefix: str = "est",
    figsize=(10, 5),
):
    """
    Plot two joint-space signals on the same axes for comparison.

    Args:
        log_t: List of time values.
        series_ref: Reference signal with shape (n_steps, nq).
        series_est: Estimated signal with shape (n_steps, nq).
        joint_indices: Which joint indices to plot (default all joints).
    """
    log_t = np.asarray(log_t)
    series_ref = np.asarray(series_ref)
    series_est = np.asarray(series_est)

    if series_ref.ndim != 2 or series_est.ndim != 2:
        raise ValueError(
            f"series_ref and series_est must both have shape (n_steps, nq), got "
            f"{series_ref.shape} and {series_est.shape}"
        )
    if series_ref.shape != series_est.shape:
        raise ValueError(
            f"series_ref and series_est must have the same shape, got "
            f"{series_ref.shape} and {series_est.shape}"
        )
    if log_t.shape[0] != series_ref.shape[0]:
        raise ValueError(
            f"log_t and signals must have the same length, got {log_t.shape[0]} and "
            f"{series_ref.shape[0]}"
        )

    if joint_indices is None:
        joint_indices = list(range(series_ref.shape[1]))

    plt.figure(figsize=figsize)
    for j in joint_indices:
        plt.plot(log_t, series_ref[:, j], "--", label=f"{ref_prefix} {j}")
        plt.plot(log_t, series_est[:, j], label=f"{est_prefix} {j}")
    plt.grid()
    plt.legend()
    plt.show()


def plot_joint_trajectory_comparison_subplots(
    log_t: list,
    series_ref,
    series_est,
    joint_indices: list = None,
    ref_label: str = "tau_e",
    est_label: str = "tau_hat_e",
    ncols: int = 2,
    figsize_per_row=(12, 3),
):
    """
    Plot joint-wise comparisons in separate subplots.

    Args:
        log_t: List of time values.
        series_ref: Reference signal with shape (n_steps, nq).
        series_est: Estimated signal with shape (n_steps, nq).
        joint_indices: Which joint indices to plot (default all joints).
    """
    log_t = np.asarray(log_t)
    series_ref = np.asarray(series_ref)
    series_est = np.asarray(series_est)

    if series_ref.ndim != 2 or series_est.ndim != 2:
        raise ValueError(
            f"series_ref and series_est must both have shape (n_steps, nq), got "
            f"{series_ref.shape} and {series_est.shape}"
        )
    if series_ref.shape != series_est.shape:
        raise ValueError(
            f"series_ref and series_est must have the same shape, got "
            f"{series_ref.shape} and {series_est.shape}"
        )
    if log_t.shape[0] != series_ref.shape[0]:
        raise ValueError(
            f"log_t and signals must have the same length, got {log_t.shape[0]} and "
            f"{series_ref.shape[0]}"
        )

    if joint_indices is None:
        joint_indices = list(range(series_ref.shape[1]))

    nplots = len(joint_indices)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, j in zip(axes, joint_indices):
        ax.plot(log_t, series_ref[:, j], "--", label=ref_label)
        ax.plot(log_t, series_est[:, j], label=est_label)
        ax.set_title(f"Joint {j}")
        ax.grid()
        ax.legend()

    for ax in axes[nplots:]:
        ax.axis("off")

    axes[(nrows - 1) * ncols].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
