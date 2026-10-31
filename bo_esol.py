"""
Pool-based Bayesian optimization for molecular solubility (ESOL / Delaney).

Task:    find the most water-soluble molecules (highest logS) among those with
         MW >= --mw-min, using as few "lab measurements" (oracle calls) as possible.
Setup:   retrospective virtual screening. Labels exist but stay hidden; a strategy
         only sees a molecule's label after it "measures" it.
Design:  MW is computed from structure for free, so it is a known feasibility
         filter applied to the pool up front. This is plain BO on a filtered
         domain, not constrained BO (no constraint is modeled).

Strategies (identical initial set per seed):
    random : uniform pick from the unmeasured pool
    greedy : argmax of GP posterior mean (exploitation only)
    ei     : argmax of Log Expected Improvement (exploration + exploitation)

Surrogate: GP with Tanimoto kernel on 2048-bit Morgan fingerprints (radius 2).
Metric:    top-k recall = fraction of the true top-k feasible molecules measured so far.
           Molecules tied with the k-th value are included in the target set.
"""
import argparse
import json
import os
import time
import warnings

import gpytorch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.exceptions import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

DATA_URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
TARGET = "measured log solubility in mols per litre"
MW = "Molecular Weight"
STRATEGIES = ["random", "greedy", "ei"]
COLORS = {"random": "C1", "greedy": "C2", "ei": "C0"}


# ---------------------------------------------------------------- data
def load_data(path, mw_min):
    df = pd.read_csv(path).dropna(subset=["smiles", TARGET])
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    n_invalid = sum(m is None for m in mols)
    df = df[[m is not None for m in mols]].copy()
    mols = [m for m in mols if m is not None]

    if MW not in df.columns:
        df[MW] = [Descriptors.MolWt(m) for m in mols]
    # Morgan fingerprints ignore stereo, so dedup on non-isomeric SMILES:
    # stereoisomers would otherwise be identical inputs with different labels.
    df["key"] = [Chem.MolToSmiles(m, isomericSmiles=False) for m in mols]
    n_before = len(df)
    df = df.groupby("key", as_index=False).agg({TARGET: "mean", MW: "first"})
    n_dups = n_before - len(df)

    df = df[df[MW] >= mw_min].reset_index(drop=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in df["key"]])
    keep = fps.sum(1) > 0                      # all-zero fingerprint breaks Tanimoto
    df, fps = df[keep].reset_index(drop=True), fps[keep]

    X = torch.tensor(fps, dtype=torch.double)
    y = torch.tensor(df[TARGET].values, dtype=torch.double)
    info = {"invalid_smiles": n_invalid, "duplicates_merged": n_dups,
            "empty_fingerprints": int((~keep).sum()), "pool_size": len(df)}
    return df, X, y, info


# ---------------------------------------------------------------- model
class TanimotoKernel(gpytorch.kernels.Kernel):
    """k(a,b) = <a,b> / (|a|^2 + |b|^2 - <a,b>). PSD on binary vectors."""
    is_stationary = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            dot = (x1 * x2).sum(-1)
            return dot / ((x1 ** 2).sum(-1) + (x2 ** 2).sum(-1) - dot).clamp_min(1e-8)
        dot = x1 @ x2.transpose(-2, -1)
        n1 = (x1 ** 2).sum(-1, keepdim=True)
        n2 = (x2 ** 2).sum(-1, keepdim=True).transpose(-2, -1)
        return dot / (n1 + n2 - dot).clamp_min(1e-8)


def make_gp(X, Y):
    return SingleTaskGP(X, Y, covar_module=gpytorch.kernels.ScaleKernel(TanimotoKernel()),
                        outcome_transform=Standardize(m=1))   # posterior in raw units


def fit_gp(X, Y, stats):
    gp = make_gp(X, Y)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", OptimizationWarning)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        if any(issubclass(x.category, OptimizationWarning) for x in w):
            stats["fit_warnings"] += 1
    except ModelFittingError:
        stats["fit_failures"] += 1
        gp = make_gp(X, Y)                     # fresh GP with default hyperparameters
    gp.eval()
    return gp


# ---------------------------------------------------------------- loop
def run(strategy, seed, X, y, top, n_init, budget, stats):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)                    # BoTorch fit retries sample from priors
    n = len(X)
    measured = [int(i) for i in rng.choice(n, n_init, replace=False)]
    recall = [len(top.intersection(measured)) / len(top)]

    while len(measured) < budget:
        pool = np.setdiff1d(np.arange(n), measured)
        if strategy == "random":
            nxt = int(rng.choice(pool))
        else:
            Xm, Ym = X[measured], y[measured].unsqueeze(-1)
            gp = fit_gp(Xm, Ym, stats)
            acq = PosteriorMean(gp) if strategy == "greedy" else LogExpectedImprovement(gp, best_f=Ym.max())
            with torch.no_grad():
                scores = acq(X[pool].unsqueeze(1))   # (n_pool, q=1, d) -> (n_pool,)
            nxt = int(pool[scores.argmax()])
        measured.append(nxt)
        recall.append(len(top.intersection(measured)) / len(top))
    return np.array(recall)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default=DATA_URL, help="URL or local path to ESOL csv")
    p.add_argument("--mw-min", type=float, default=250.0, help="feasibility filter; 0 disables")
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--budget", type=int, default=50, help="total measurements incl. initial set")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    df, X, y, info = load_data(a.data, a.mw_min)
    n = len(y)
    if not (1 <= a.n_init < a.budget <= n):
        p.error(f"need 1 <= n_init < budget <= pool size ({n})")
    if not (1 <= a.top_k <= n):
        p.error(f"need 1 <= top_k <= pool size ({n})")
    if a.seeds < 2:
        p.error("need seeds >= 2 for standard errors")

    threshold = torch.topk(y, a.top_k).values.min()
    top = set(torch.nonzero(y >= threshold).flatten().tolist())   # includes ties
    os.makedirs(a.out, exist_ok=True)
    print(f"Pool (MW >= {a.mw_min:g}): {n} molecules | merged {info['duplicates_merged']} duplicates, "
          f"dropped {info['invalid_smiles']} invalid SMILES")
    print(f"Target: top-{len(top)} molecules, logS >= {threshold:.2f} | "
          f"budget {a.budget} = {a.budget / n:.0%} of pool | {a.seeds} seeds")

    curves, stats_all, summary = {}, {}, []
    for s in a.strategies:
        stats = {"fit_failures": 0, "fit_warnings": 0}
        t0 = time.time()
        curves[s] = np.stack([run(s, seed, X, y, top, a.n_init, a.budget, stats)
                              for seed in range(a.seeds)])
        final, init = curves[s][:, -1], curves[s][:, 0]
        se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
        summary.append({
            "strategy": s,
            "recall_mean": final.mean(), "recall_se": se(final),
            "gain_over_init_mean": (final - init).mean(),          # excludes lucky init hits
            "seeds_with_new_hit": f"{int(((final - init) > 0).sum())}/{a.seeds}",
            "gp_fit_failures": stats["fit_failures"], "gp_fit_warnings": stats["fit_warnings"],
            "seconds": round(time.time() - t0, 1),
        })
        stats_all[s] = stats
        print(f"  {s:7s} recall {final.mean():.3f} ± {se(final):.3f}  ({summary[-1]['seconds']}s)")

    table = pd.DataFrame(summary)
    print("\n" + table.to_string(index=False, float_format="%.3f"))
    table.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    np.savez(os.path.join(a.out, "curves.npz"), **curves)
    with open(os.path.join(a.out, "run_info.json"), "w") as f:
        json.dump({"args": vars(a), "data": info, "top_set_size": len(top),
                   "logS_threshold": float(threshold)}, f, indent=2)

    steps = np.arange(a.n_init, a.budget + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s, c in curves.items():
        m, se_c = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax.plot(steps, m, color=COLORS[s], label=s)
        ax.fill_between(steps, m - se_c, m + se_c, color=COLORS[s], alpha=0.2)
    ax.set_xlabel("molecules measured (oracle calls)")
    ax.set_ylabel(f"top-{len(top)} recall")
    ax.set_title(f"ESOL, MW >= {a.mw_min:g}: mean ± SE over {a.seeds} seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "recall.png"), dpi=150)
    print(f"\nSaved {a.out}/recall.png, summary.csv, curves.npz, run_info.json")


if __name__ == "__main__":
    main()
