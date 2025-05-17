"""
Results
"""

import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import sparse as sp


def read_losses(
    homogeneous: bool,
    n_years: int,
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    n_sim=1_000_000,
    dir="Outputs",
) -> npt.NDArray[np.float64]:

    loss_df = pd.read_parquet(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}.parquet"
    )

    loss_per_sim = np.array(loss_df.groupby("simulation")["total_loss"].sum())

    n_sim_missing = n_sim - len(loss_per_sim)
    loss_per_sim = np.append(loss_per_sim, [0] * n_sim_missing)

    return loss_per_sim

def read_cyclone_counts(
        homogeneous: bool,
        n_years: int,
        loc: bool = False,
        p_loc: float = 0.5,
        intensity: bool = False,
        damage: bool = False,
        n_sim: int = 1_000_000,
        dir: str = "Outputs",
):
    
    cyclones_df  = pd.read_csv(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}_cyclone_counts.csv"
    )

    cyclone_count_per_sim = np.array(cyclones_df.filter(like = "Number of cyclones").sum(1))

    return cyclone_count_per_sim


def plot_return_period_curves(
    losses_per_sim: list, legend_labels: dict, n_sim=1_000_000, rp=250
):

    exceed_freq = np.linspace(1 / n_sim, 1, n_sim)
    return_period = 1 / exceed_freq[::-1]
    max_idx = int((1 - 1 / rp) * n_sim + 1)

    plt.figure(figsize=(7.5, 7.5 / 1.6))

    for loss_per_sim in losses_per_sim:
        sorted_values = np.sort(loss_per_sim)

        for key, value in legend_labels.items():
            if np.array_equal(loss_per_sim, value):
                label = key

        plt.plot(return_period[:max_idx], sorted_values[:max_idx], label=label)

    plt.xlabel("Return period")
    plt.ylabel("Loss (USD)")
    plt.grid(color="whitesmoke")
    plt.legend(loc="upper left")
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    return None


def risk_measures(losses_per_sim: list, column_labels: dict, units: int) -> pd.DataFrame:

    df = pd.DataFrame()

    for loss_per_sim in losses_per_sim:
        mean = loss_per_sim.mean()
        sd = loss_per_sim.std()
        VaR_50 = np.percentile(loss_per_sim, 50)
        VaR_75 = np.percentile(loss_per_sim, 75)
        VaR_95 = np.percentile(loss_per_sim, 95)
        VaR_99 = np.percentile(loss_per_sim, 99)
        VaR_995 = np.percentile(loss_per_sim, 99.5)
        TVaR_99 = np.mean(loss_per_sim[loss_per_sim > VaR_99])
        TVaR_995 = np.mean(loss_per_sim[loss_per_sim > VaR_995])

        all_measures = [
            VaR_50,
            VaR_75,
            VaR_95,
            VaR_99,
            VaR_995,
            mean,
            sd,
            TVaR_99,
            TVaR_995,
        ]

        for key, value in column_labels.items():
            if np.array_equal(loss_per_sim, value):
                colname = key

        df[colname] = all_measures

    df_styled = df.style.format(lambda x: f"{x/units:,.0f}")

    return df_styled
