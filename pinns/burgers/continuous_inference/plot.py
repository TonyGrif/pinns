"""Visualization utilities for Burgers' equation predictions"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pinns.burgers.continuous_inference.burgers import BurgersPINN


@dataclass
class PlotConfig:
    """Data needed to generate plots during training

    Attributes:
        X_star: All evaluation points
        Exact: Exact solution
        X: Spatial meshgrid
        T: Temporal meshgrid
        x: Spatial grid
        t: Temporal grid
        tag: Label string used in filenames and titles
        out_dir: Directory to save plots
    """

    X_star: np.ndarray
    Exact: np.ndarray
    X: np.ndarray
    T: np.ndarray
    x: np.ndarray
    t: np.ndarray
    tag: str
    out_dir: Path


def plot_heatmap(
    model: BurgersPINN,
    epoch: int,
    X_star: np.ndarray,
    Exact: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    tag: str,
    out_dir: Path,
) -> None:
    """Side-by-side heatmap of the predicted and exact state

    Args:
        model: BurgersPINN instance
        epoch: Current training epoch
        X_star: All evaluation points
        Exact: Exact solution
        X: Spatial meshgrid
        T: Temporal meshgrid
        tag: Label string
        out_dir: Directory to save the figure
    """
    u_pred = model.predict(X_star).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field, title in zip(
        axes,
        [u_pred, Exact],
        [f"Predicted ({tag.upper()})", "Exact"],
    ):
        im = ax.imshow(
            field.T,
            interpolation="nearest",
            cmap="rainbow",
            extent=(T.min(), T.max(), X.min(), X.max()),
            origin="lower",
            aspect="auto",
        )
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"State at Epoch {epoch}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tag}_heatmap_{epoch:05d}.png", dpi=300)
    plt.close(fig)


def plot_error(
    model: BurgersPINN,
    epoch: int,
    X_star: np.ndarray,
    Exact: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    tag: str,
    out_dir: Path,
) -> None:
    """Save an absolute error heatmap

    Args:
        model: BurgersPINN instance
        epoch: Current training epoch
        X_star: All evaluation points
        Exact: Exact solution
        X: Spatial meshgrid
        T: Temporal meshgrid
        tag: Label string
        out_dir: Directory to save the figure
    """
    u_pred = model.predict(X_star).reshape(X.shape)
    error = np.abs(Exact - u_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        error.T,
        interpolation="nearest",
        cmap="hot",
        extent=(T.min(), T.max(), X.min(), X.max()),
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title(f"Absolute Error - {tag.upper()} at Epoch {epoch}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tag}_error_{epoch:05d}.png", dpi=300)
    plt.close(fig)


def plot_slices(
    model: BurgersPINN,
    epoch: int,
    x: np.ndarray,
    t: np.ndarray,
    Exact: np.ndarray,
    tag: str,
    out_dir: Path,
) -> None:
    """Line plot slices at t=0.25, 0.50, 0.75

    Args:
        model: BurgersPINN instance
        epoch: Current training epoch
        x: Spatial grid
        t: Temporal grid
        Exact: Exact solution
        tag: Label string
        out_dir: Directory to save the figure
    """
    t_flat = t.flatten()
    snap_fracs = [0.25, 0.50, 0.75]
    snap_indices = [int(f * (len(t_flat) - 1)) for f in snap_fracs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, idx, frac in zip(axes, snap_indices, snap_fracs):
        t_val = t_flat[idx]
        t_col = np.full_like(x, t_val)
        X_slice = np.hstack([x, t_col])
        u_pred = model.predict(X_slice).flatten()

        ax.plot(x.flatten(), Exact[idx, :], "b-", linewidth=2, label="Exact")
        ax.plot(x.flatten(), u_pred, "r--", linewidth=2, label=f"{tag.upper()}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u(t, x)$")
        ax.set_title(f"$t = {frac:.2f}$")
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.legend(frameon=False)

    fig.suptitle(f"Solution Slices at Epoch {epoch}")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{tag}_slices_{epoch:05d}.png", dpi=300)
    plt.close(fig)


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
