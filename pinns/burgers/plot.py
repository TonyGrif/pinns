"""Shared visualization utilities for Burgers' equation datasets"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_exact(
    x: np.ndarray,
    t: np.ndarray,
    Exact: np.ndarray,
    out_dir: Path,
) -> None:
    """Heatmap of the exact solution

    Args:
        x: Spatial grid
        t: Temporal grid
        Exact: Exact solution
        out_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Exact.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=(t.min(), t.max(), x.min(), x.max()),
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("Exact State")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exact.png", dpi=300)
    plt.close(fig)


def plot_dataset(
    x: np.ndarray,
    t: np.ndarray,
    Exact: np.ndarray,
    X_u_train: np.ndarray,
    out_dir: Path,
) -> None:
    """Heatmap of the exact solution with training (IC/BC) points

    Args:
        x: Spatial grid
        t: Temporal grid
        Exact: Exact solution
        X_u_train: Training points
        out_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Exact.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=(t.min(), t.max(), x.min(), x.max()),
        origin="lower",
        aspect="auto",
    )
    ax.scatter(X_u_train[:, 1], X_u_train[:, 0], c="k", s=12, label="Training Points")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("Exact State with Training Points")
    ax.legend(frameon=False)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "dataset.png", dpi=300)
    plt.close(fig)


def plot_collocation(
    x: np.ndarray,
    t: np.ndarray,
    Exact: np.ndarray,
    X_u_train: np.ndarray,
    X_f_train: np.ndarray,
    out_dir: Path,
) -> None:
    """Heatmap of the exact solution with training and collocation points

    Args:
        x: Spatial grid
        t: Temporal grid
        Exact: Exact solution
        X_u_train: Training points
        X_f_train: Collocation points
        out_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Exact.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=(t.min(), t.max(), x.min(), x.max()),
        origin="lower",
        aspect="auto",
    )
    ax.scatter(
        X_f_train[:, 1],
        X_f_train[:, 0],
        c="w",
        s=4,
        alpha=0.4,
        label="Collocation Points",
    )
    ax.scatter(X_u_train[:, 1], X_u_train[:, 0], c="k", s=12, label="Training Points")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("Exact State with Training and Collocation Points")
    ax.legend(frameon=False)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "collocation.png", dpi=300)
    plt.close(fig)
