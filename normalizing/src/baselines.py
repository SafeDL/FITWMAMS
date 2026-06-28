"""Density baselines for highD tail c0 features."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.mixture import GaussianMixture

from .data import split_indices
from .model import build_realnvp_flow


logger = logging.getLogger(__name__)


def _regularized_cov(x: np.ndarray, ridge: float) -> np.ndarray:
    cov = np.cov(np.asarray(x, dtype=np.float64), rowvar=False)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    cov = cov + float(ridge) * np.eye(cov.shape[0], dtype=np.float64)
    return cov


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("Gaussian covariance is not positive definite")
    inv = np.linalg.inv(cov)
    centered = x - mean.reshape(1, -1)
    mahal = np.einsum("bi,ij,bj->b", centered, inv, centered)
    dim = x.shape[1]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + mahal)


def gaussian_baseline_nll(
    x_train: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    ridge: float,
) -> dict[str, float]:
    mean = np.mean(x_train, axis=0)
    cov = _regularized_cov(x_train, ridge)
    return {
        split: float(-np.mean(_gaussian_logpdf(values, mean, cov)))
        for split, values in splits.items()
        if len(values) > 0
    }


def gmm_baseline_nll(
    x_train: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    n_components: int,
    reg_covar: float,
    seed: int,
) -> dict[str, float]:
    n_components = min(int(n_components), max(1, len(x_train) // 20))
    model = GaussianMixture(
        n_components=max(1, n_components),
        covariance_type="full",
        reg_covar=float(reg_covar),
        random_state=int(seed),
        max_iter=300,
        n_init=2,
    )
    model.fit(x_train)
    return {
        split: float(-np.mean(model.score_samples(values)))
        for split, values in splits.items()
        if len(values) > 0
    }


def _rank_to_gaussian_train(x_train: np.ndarray) -> np.ndarray:
    z = np.zeros_like(x_train, dtype=np.float64)
    n = x_train.shape[0]
    for j in range(x_train.shape[1]):
        ranks = rankdata(x_train[:, j], method="average")
        u = np.clip((ranks - 0.5) / max(n, 1), 1.0e-5, 1.0 - 1.0e-5)
        z[:, j] = norm.ppf(u)
    return z


def _empirical_to_gaussian(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    n = train.shape[0]
    for j in range(train.shape[1]):
        sorted_train = np.sort(train[:, j])
        ranks = np.searchsorted(sorted_train, values[:, j], side="right")
        u = np.clip((ranks + 0.5) / (n + 1.0), 1.0e-5, 1.0 - 1.0e-5)
        out[:, j] = norm.ppf(u)
    return out


def gaussian_copula_baseline_nll(
    x_train: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    ridge: float,
) -> dict[str, float]:
    z_train = _rank_to_gaussian_train(x_train)
    corr = _regularized_cov(z_train, ridge)
    std = np.sqrt(np.maximum(np.diag(corr), 1.0e-12))
    corr = corr / np.outer(std, std)
    corr = corr + float(ridge) * np.eye(corr.shape[0], dtype=np.float64)
    out: dict[str, float] = {}
    for split, values in splits.items():
        if len(values) == 0:
            continue
        z = _empirical_to_gaussian(x_train, values)
        log_joint_z = _gaussian_logpdf(z, np.zeros(z.shape[1]), corr)
        log_ind_z = np.sum(norm.logpdf(z), axis=1)
        log_ind_x = np.sum(norm.logpdf(values), axis=1)
        log_prob = log_joint_z - log_ind_z + log_ind_x
        out[split] = float(-np.mean(log_prob))
    return out


def _torch_loaders(x: np.ndarray, arrays: dict[str, np.ndarray], batch_size: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for split in ("train", "val", "test"):
        idx = split_indices(arrays, split)
        tensor = torch.from_numpy(x[idx]).float()
        loaders[split] = DataLoader(
            TensorDataset(tensor),
            batch_size=int(batch_size),
            shuffle=(split == "train"),
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def train_realnvp_baseline(
    arrays: dict[str, np.ndarray],
    *,
    cfg: dict[str, Any],
    repo_root: str | Path,
    output_dir: str | Path,
    device,
) -> dict[str, Any]:
    import torch

    x = np.asarray(arrays["features_normalized"], dtype=np.float32)
    flow = build_realnvp_flow(
        num_features=x.shape[1],
        model_cfg=dict(cfg.get("model", {})),
        repo_root=repo_root,
    ).to(device)
    train_cfg = dict(cfg.get("training", {}))
    loaders = _torch_loaders(x, arrays, int(train_cfg.get("batch_size", 256)))
    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=float(train_cfg.get("learning_rate", 5.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    max_epochs = int(train_cfg.get("max_epochs", 40))
    patience = int(train_cfg.get("patience", 8))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        flow.train()
        total = 0.0
        total_n = 0
        for (batch_x,) in loaders["train"]:
            batch_x = batch_x.to(device, non_blocking=True)
            loss = -flow.log_prob(batch_x).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
            optimizer.step()
            n = int(batch_x.shape[0])
            total += float(loss.detach().cpu()) * n
            total_n += n
        train_nll = total / max(total_n, 1)
        val_nll = _eval_unconditional_flow(flow, loaders["val"], device)
        logger.info(
            "RealNVP baseline epoch %03d train_nll=%.4f val_nll=%.4f",
            epoch,
            train_nll,
            val_nll,
        )
        if val_nll < best_val:
            best_val = val_nll
            best_state = {key: value.detach().cpu() for key, value in flow.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        flow.load_state_dict(best_state)
    metrics = {
        split: _eval_unconditional_flow(loader=loaders[split], flow=flow, device=device)
        for split in ("train", "val", "test")
    }
    ckpt = Path(output_dir) / "checkpoints" / "baseline_realnvp.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "model_cfg": dict(cfg.get("model", {})),
            "metrics": metrics,
        },
        ckpt,
    )
    return {"nll": metrics, "checkpoint": str(ckpt)}


def _eval_unconditional_flow(flow, loader, device) -> float:
    import torch

    flow.eval()
    total = 0.0
    total_n = 0
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            nll = -flow.log_prob(batch_x)
            n = int(batch_x.shape[0])
            total += float(nll.sum().detach().cpu())
            total_n += n
    return total / max(total_n, 1)


def fit_density_baselines(
    arrays: dict[str, np.ndarray],
    *,
    cfg: dict[str, Any],
    repo_root: str | Path,
    output_dir: str | Path,
    device,
) -> dict[str, Any]:
    x = np.asarray(arrays["features_normalized"], dtype=np.float32)
    train_idx = split_indices(arrays, "train")
    split_values = {
        split: x[split_indices(arrays, split)]
        for split in ("train", "val", "test")
    }
    baseline_cfg = dict(cfg)
    out: dict[str, Any] = {}
    out["gaussian"] = {
        "nll": gaussian_baseline_nll(
            x[train_idx],
            split_values,
            ridge=float(baseline_cfg.get("gaussian", {}).get("ridge", 1.0e-4)),
        )
    }
    out["gmm"] = {
        "nll": gmm_baseline_nll(
            x[train_idx],
            split_values,
            n_components=int(baseline_cfg.get("gmm", {}).get("n_components", 8)),
            reg_covar=float(baseline_cfg.get("gmm", {}).get("reg_covar", 1.0e-5)),
            seed=int(baseline_cfg.get("seed", 42)),
        )
    }
    out["copula"] = {
        "nll": gaussian_copula_baseline_nll(
            x[train_idx],
            split_values,
            ridge=float(baseline_cfg.get("copula", {}).get("ridge", 1.0e-4)),
        )
    }
    if bool(baseline_cfg.get("realnvp", {}).get("enabled", True)):
        out["unconditional_realnvp"] = train_realnvp_baseline(
            arrays,
            cfg=dict(baseline_cfg.get("realnvp", {})),
            repo_root=repo_root,
            output_dir=output_dir,
            device=device,
        )
    return out
