"""Burgers' equation continuous inference"""

import logging
from pathlib import Path

import numpy as np
import torch

from pinns.burgers.plot import PlotConfig, plot_error, plot_heatmap, plot_slices
from pinns.net import Net

logger = logging.getLogger(__name__)


class BurgersPINN:
    """Wrapper for Burgers' equation continuous inference

    Supports two training modes:
      * fit_nn: standard supervised training on data only
      * fit_pinn: adds physics residual loss
    """

    def __init__(
        self,
        net: Net,
        lb: np.ndarray | list,
        ub: np.ndarray | list,
        nu: float,
    ) -> None:
        """Constructor for BurgersPINN

        Args:
            net: Configured network approximating the solution `u(x, t)`
            lb: Lower bounds of the input domain
            ub: Upper bounds of the input domain
            nu: Viscosity coefficient
        """
        self.net = net
        self.nu = nu

        device = next(net.parameters()).device
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        self._plot_config: PlotConfig | None = None

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        device = next(self.net.parameters()).device
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def _set_plot_config(
        self,
        X_star: np.ndarray,
        Exact: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        x: np.ndarray,
        t: np.ndarray,
        tag: str,
        out_dir: Path,
    ) -> None:
        self._plot_config = PlotConfig(
            X_star=X_star, Exact=Exact, X=X, T=T, x=x, t=t, tag=tag, out_dir=out_dir
        )

    def normalize(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [-1, 1] using stored domain bounds

        Args:
            X: Input tensor

        Returns:
            Normalized input tensor
        """
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize X and pass through the network

        Args:
            X: Input tensor

        Returns:
            Predicted state solution `u`
        """
        return self.net(self.normalize(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run prediction and return a numpy array

        Args:
            X: Input array

        Returns:
            Predicted solution `u`
        """
        with torch.no_grad():
            return self.forward(self._to_tensor(X)).cpu().numpy()

    def residual(self, X_f: torch.Tensor) -> torch.Tensor:
        """Compute the Burgers' equation residual: `f = u_t + u*u_x - nu*u_xx`

        Args:
            X_f: Collocation points with `requires_grad=True`

        Returns:
            Residual tensor
        """
        u = self.forward(X_f)

        grads = torch.autograd.grad(
            u,
            X_f,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x,
            X_f,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0][:, 0:1]

        return u_t + u * u_x - self.nu * u_xx

    def fit_nn(
        self,
        X_u: np.ndarray,
        u: np.ndarray,
        epochs: int,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train using data loss only

        Args:
            X_u: Boundary/IC data points
            u: Observed solution values
            epochs: Number of gradient steps
            lr: Adam learning rate

        Returns:
            Loss history
        """
        X_u_t = self._to_tensor(X_u)
        u_t = self._to_tensor(u)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        history: list[float] = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((self.forward(X_u_t) - u_t) ** 2)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if epoch % 100 == 0:
                logger.info("[NN]   epoch %5d  loss = %.6e", epoch, loss.item())
                if self._plot_config is not None:
                    cfg = self._plot_config
                    plot_heatmap(
                        self,
                        epoch,
                        cfg.X_star,
                        cfg.Exact,
                        cfg.X,
                        cfg.T,
                        cfg.tag,
                        cfg.out_dir,
                    )
                    plot_error(
                        self,
                        epoch,
                        cfg.X_star,
                        cfg.Exact,
                        cfg.X,
                        cfg.T,
                        cfg.tag,
                        cfg.out_dir,
                    )
                    plot_slices(
                        self, epoch, cfg.x, cfg.t, cfg.Exact, cfg.tag, cfg.out_dir
                    )

        return history

    def fit_pinn(
        self,
        X_u: np.ndarray,
        u: np.ndarray,
        X_f: np.ndarray,
        epochs: int,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train using data loss and physics residual loss

        Args:
            X_u: Boundary/IC data points
            u: Observed solution values
            X_f: Collocation points
            epochs: Number of gradient steps
            lr: Adam learning rate

        Returns:
            Loss history
        """
        X_u_t = self._to_tensor(X_u)
        u_t = self._to_tensor(u)
        X_f_t = self._to_tensor(X_f)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        history: list[float] = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            X_f_r = X_f_t.clone().detach().requires_grad_(True)

            data_loss = torch.mean((self.forward(X_u_t) - u_t) ** 2)
            phys_loss = torch.mean(self.residual(X_f_r) ** 2)
            loss = data_loss + phys_loss

            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if epoch % 100 == 0:
                logger.info("[PINN] epoch %5d  loss = %.6e", epoch, loss.item())
                if self._plot_config is not None:
                    cfg = self._plot_config
                    plot_heatmap(
                        self,
                        epoch,
                        cfg.X_star,
                        cfg.Exact,
                        cfg.X,
                        cfg.T,
                        cfg.tag,
                        cfg.out_dir,
                    )
                    plot_error(
                        self,
                        epoch,
                        cfg.X_star,
                        cfg.Exact,
                        cfg.X,
                        cfg.T,
                        cfg.tag,
                        cfg.out_dir,
                    )
                    plot_slices(
                        self, epoch, cfg.x, cfg.t, cfg.Exact, cfg.tag, cfg.out_dir
                    )

        return history
