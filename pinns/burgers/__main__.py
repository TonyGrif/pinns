"""Burgers' equation training script; Run with: `python -m pinns.burgers`"""

import logging
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
from scipy.stats.qmc import LatinHypercube

from pinns.burgers.burgers import BurgersPINN
from pinns.burgers.plot import plot_collocation, plot_dataset, plot_exact
from pinns.net import Net

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NU = 0.01 / np.pi
N_U = 100
N_F = 10000
EPOCHS = 50000
LR = 1e-3


def main() -> None:
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Load data
    data = scipy.io.loadmat("datasets/burgers_shock.mat")
    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    Exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x, t)
    X_star = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    u_star = Exact.flatten()[:, None]

    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)

    # Boundary and initial condition points
    xx1 = np.hstack([X[0:1, :].T, T[0:1, :].T])
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack([X[:, 0:1], T[:, 0:1]])
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack([X[:, -1:], T[:, -1:]])
    uu3 = Exact[:, -1:]

    X_u_all = np.vstack([xx1, xx2, xx3])
    u_all = np.vstack([uu1, uu2, uu3])
    idx = np.random.choice(X_u_all.shape[0], N_U, replace=False)
    X_u_train = X_u_all[idx]
    u_train = u_all[idx]

    # Collocation points
    sampler = LatinHypercube(d=2)
    X_f_train = lb + (ub - lb) * sampler.random(n=N_F)
    X_f_train = np.vstack([X_f_train, X_u_all])

    # Dataset diagnostics
    plot_exact(x, t, Exact, out_dir=Path("plots"))
    plot_dataset(x, t, Exact, X_u_train, out_dir=Path("plots"))
    plot_collocation(x, t, Exact, X_u_train, X_f_train, out_dir=Path("plots"))

    # Standard NN
    logger.info("Training: Standard NN")
    net_nn = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_nn = BurgersPINN(net=net_nn, lb=lb, ub=ub, nu=NU)
    # TODO: Make set config method to public
    model_nn._set_plot_config(
        X_star, Exact, X, T, x, t, tag="nn", out_dir=Path("plots/nn")
    )

    t0 = time.time()
    model_nn.fit_nn(X_u_train, u_train, epochs=EPOCHS, lr=LR)
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_nn = model_nn.predict(X_star)
    error_nn = np.linalg.norm(u_star - u_pred_nn, 2) / np.linalg.norm(u_star, 2)

    # PINN
    logger.info("Training: PINN")
    net_pinn = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_pinn = BurgersPINN(net=net_pinn, lb=lb, ub=ub, nu=NU)
    model_pinn._set_plot_config(
        X_star, Exact, X, T, x, t, tag="pinn", out_dir=Path("plots/pinn")
    )

    t0 = time.time()
    model_pinn.fit_pinn(X_u_train, u_train, X_f_train, epochs=EPOCHS, lr=LR)
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_pinn = model_pinn.predict(X_star)
    error_pinn = np.linalg.norm(u_star - u_pred_pinn, 2) / np.linalg.norm(u_star, 2)

    # Noisy PINN - 1%
    logger.info("Training: Noisy PINN (1%% noise)")
    net_noisy1 = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_noisy1 = BurgersPINN(net=net_noisy1, lb=lb, ub=ub, nu=NU)
    model_noisy1._set_plot_config(
        X_star, Exact, X, T, x, t, tag="noisy1", out_dir=Path("plots/noisy1")
    )

    t0 = time.time()
    model_noisy1.fit_noisy_pinn(
        X_u_train, u_train, X_f_train, epochs=EPOCHS, noise_level=0.01, lr=LR
    )
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_noisy1 = model_noisy1.predict(X_star)
    error_noisy1 = np.linalg.norm(u_star - u_pred_noisy1, 2) / np.linalg.norm(u_star, 2)

    # Noisy PINN - 10%
    logger.info("Training: Noisy PINN (10%% noise)")
    net_noisy10 = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_noisy10 = BurgersPINN(net=net_noisy10, lb=lb, ub=ub, nu=NU)
    model_noisy10._set_plot_config(
        X_star, Exact, X, T, x, t, tag="noisy10", out_dir=Path("plots/noisy10")
    )

    t0 = time.time()
    model_noisy10.fit_noisy_pinn(
        X_u_train, u_train, X_f_train, epochs=EPOCHS, noise_level=0.10, lr=LR
    )
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_noisy10 = model_noisy10.predict(X_star)
    error_noisy10 = np.linalg.norm(u_star - u_pred_noisy10, 2) / np.linalg.norm(
        u_star, 2
    )

    # Results
    logger.info("L2 error  Standard NN      : %.4e", error_nn)
    logger.info("L2 error  PINN             : %.4e", error_pinn)
    logger.info("L2 error  Noisy PINN  1%%  : %.4e", error_noisy1)
    logger.info("L2 error  Noisy PINN 10%%  : %.4e", error_noisy10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
