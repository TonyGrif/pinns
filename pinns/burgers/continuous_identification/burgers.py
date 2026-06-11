"""Burgers' equation continuous identification"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pinns.burgers.plot import PlotConfig, plot_error, plot_heatmap, plot_slices
from pinns.net import Net

logger = logging.getLogger(__name__)


class BurgersIdentificationPINN:
    """Wrapper for Burgers' equation continuous identification"""

    def __init__(
        self,
        net: Net,
        lb: np.ndarray | list,
        ub: np.ndarray | list,
    ) -> None:
        """Constructor for BurgersIdentificationPINN

        Args:
            net: Configured network
            lb: Lower bounds of the input domain
            ub: Upper bounds of the input domain
        """
        self.net = net

        device = next(net.parameters()).device
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)

        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=device))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], device=device))

        self._plot_config: PlotConfig | None = None

    @property
    def lambda1(self) -> float:
        """Identified lambda 1"""
        return self.lambda_1.item()

    @property
    def lambda2(self) -> float:
        """Identified lambda 2"""
        return torch.exp(self.lambda_2).item()

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
            Predicted state solution
        """
        return self.net(self.normalize(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run prediction and return a numpy array

        Args:
            X: Input array

        Returns:
            Predicted solution
        """
        with torch.no_grad():
            return self.forward(self._to_tensor(X)).cpu().numpy()

    def residual(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the identified Burgers' residual

        Args:
            X: Collocation points

        Returns:
            Residual tensor
        """
        u = self.forward(X)

        grads = torch.autograd.grad(
            u,
            X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]

        u_xx = torch.autograd.grad(
            u_x,
            X,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0][:, 0:1]

        return u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx

    def fit(
        self,
        X_u: np.ndarray,
        u: np.ndarray,
        epochs: int,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train using Adam followed by L-BFGS

        Args:
            X_u: Interior data points
            u: Observed solution values
            epochs: Number of Adam gradient steps
            lr: Adam learning rate

        Returns:
            Adam loss history
        """
        X_u_t = self._to_tensor(X_u)
        u_t = self._to_tensor(u)

        params = list(self.net.parameters()) + [self.lambda_1, self.lambda_2]
        history: list[float] = []

        # Adam phase
        if epochs > 0:
            optimizer_adam = torch.optim.Adam(params, lr=lr)
            for epoch in range(epochs):
                optimizer_adam.zero_grad()

                X_r = X_u_t.clone().detach().requires_grad_(True)

                data_loss = torch.mean((self.forward(X_u_t) - u_t) ** 2)
                phys_loss = torch.mean(self.residual(X_r) ** 2)
                loss = data_loss + phys_loss

                loss.backward()
                optimizer_adam.step()
                history.append(loss.item())

                if epoch % 1000 == 0:
                    logger.info(
                        "[ID][Adam]  epoch %5d  loss = %.6e  lambda_1 = %.5f  lambda_2 = %.7f",  # noqa: E501
                        epoch,
                        loss.item(),
                        self.lambda1,
                        self.lambda2,
                    )
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

        # L-BFGS
        # Converts to float64 to match paper results
        self.net.double()
        self.lb = self.lb.double()
        self.ub = self.ub.double()
        self.lambda_1.data = self.lambda_1.data.double()
        self.lambda_2.data = self.lambda_2.data.double()
        X_u_64 = X_u_t.double()
        u_64 = u_t.double()

        params64 = list(self.net.parameters()) + [self.lambda_1, self.lambda_2]
        optimizer_lbfgs = torch.optim.LBFGS(
            params64,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=float(np.finfo(np.float64).eps),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer_lbfgs.zero_grad()
            X_r = X_u_64.clone().detach().requires_grad_(True)
            data_loss = torch.mean((self.forward(X_u_64) - u_64) ** 2)
            phys_loss = torch.mean(self.residual(X_r) ** 2)
            loss = data_loss + phys_loss
            loss.backward()
            return loss

        logger.info(
            "[ID][LBFGS] start  lambda_1 = %.5f  lambda_2 = %.7f",
            self.lambda1,
            self.lambda2,
        )
        optimizer_lbfgs.step(closure)
        logger.info(
            "[ID][LBFGS] done   lambda_1 = %.5f  lambda_2 = %.7f",
            self.lambda1,
            self.lambda2,
        )

        # Cast back to float32
        self.net.float()
        self.lb = self.lb.float()
        self.ub = self.ub.float()
        self.lambda_1.data = self.lambda_1.data.float()
        self.lambda_2.data = self.lambda_2.data.float()

        if self._plot_config is not None:
            cfg = self._plot_config
            plot_heatmap(
                self, epochs, cfg.X_star, cfg.Exact, cfg.X, cfg.T, cfg.tag, cfg.out_dir
            )
            plot_error(
                self, epochs, cfg.X_star, cfg.Exact, cfg.X, cfg.T, cfg.tag, cfg.out_dir
            )
            plot_slices(self, epochs, cfg.x, cfg.t, cfg.Exact, cfg.tag, cfg.out_dir)

        return history
