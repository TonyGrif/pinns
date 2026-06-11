"""Run with: `python -m pinns.burgers.continuous_identification`"""

import logging
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from pinns.burgers.continuous_identification.burgers import BurgersIdentificationPINN
from pinns.burgers.plot import plot_dataset, plot_exact
from pinns.net import Net

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NU = 0.01 / np.pi
N_U = 2000
ADAM_EPOCHS_NOISY = 10000
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

    # Training points sampled randomly from the interior
    idx = np.random.choice(X_star.shape[0], N_U, replace=False)
    X_u_train = X_star[idx]
    u_train = u_star[idx]

    # Dataset diagnostics
    plot_exact(x, t, Exact, out_dir=Path("plots"))
    plot_dataset(x, t, Exact, X_u_train, out_dir=Path("plots"))

    # Clean data
    logger.info("Training: Identification (clean data)")
    net_clean = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_clean = BurgersIdentificationPINN(net=net_clean, lb=lb, ub=ub)
    model_clean._set_plot_config(
        X_star, Exact, X, T, x, t, tag="id_clean", out_dir=Path("plots/id_clean")
    )

    t0 = time.time()
    model_clean.fit(X_u_train, u_train, epochs=0, lr=LR)
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_clean = model_clean.predict(X_star)
    error_u_clean = np.linalg.norm(u_star - u_pred_clean, 2) / np.linalg.norm(u_star, 2)
    error_l1_clean = np.abs(model_clean.lambda1 - 1.0) * 100
    error_l2_clean = np.abs(model_clean.lambda2 - NU) / NU * 100

    # Noisy data (1%)
    logger.info("Training: Identification (1%% noise)")
    u_train_noisy = u_train + 0.01 * np.std(u_train) * np.random.randn(*u_train.shape)

    net_noisy = Net(input_dim=2, hidden_dim=20, layers=8, output_dim=1, device=DEVICE)
    model_noisy = BurgersIdentificationPINN(net=net_noisy, lb=lb, ub=ub)
    model_noisy._set_plot_config(
        X_star, Exact, X, T, x, t, tag="id_noisy", out_dir=Path("plots/id_noisy")
    )

    t0 = time.time()
    model_noisy.fit(X_u_train, u_train_noisy, epochs=ADAM_EPOCHS_NOISY, lr=LR)
    logger.info("Training time: %.2fs", time.time() - t0)

    u_pred_noisy = model_noisy.predict(X_star)
    error_u_noisy = np.linalg.norm(u_star - u_pred_noisy, 2) / np.linalg.norm(u_star, 2)
    error_l1_noisy = np.abs(model_noisy.lambda1 - 1.0) * 100
    error_l2_noisy = np.abs(model_noisy.lambda2 - NU) / NU * 100

    # Results
    logger.info("--- Clean data ---")
    logger.info("L2 error u        : %.4e", error_u_clean)
    logger.info("Error λ₁          : %.5f%%", error_l1_clean)
    logger.info("Error λ₂          : %.5f%%", error_l2_clean)
    logger.info("--- 1%% noise ---")
    logger.info("L2 error u        : %.4e", error_u_noisy)
    logger.info("Error λ₁          : %.5f%%", error_l1_noisy)
    logger.info("Error λ₂          : %.5f%%", error_l2_noisy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
