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


def read_losses_region(
    homogeneous: bool,
    n_years: int,
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    n_sim=1_000_000,
    dir="Outputs",
):
    
    loss_df = pd.read_parquet(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}.parquet"
    )

    cols_to_keep = loss_df.columns[loss_df.columns.str.contains("simulation|loss", regex = True)]
    cols_to_keep = [col for col in cols_to_keep if col not in ["total_loss", "larger_losses"]]

    loss_df_regions = loss_df[cols_to_keep]

    loss_per_sim_region = loss_df_regions.groupby("simulation").sum()
    n_sim_missing = n_sim - loss_per_sim_region.shape[0]
    new_rows = pd.DataFrame(0, index = range(n_sim_missing), columns = loss_per_sim_region.columns)
    loss_per_sim_region = pd.concat([loss_per_sim_region, new_rows], ignore_index=True)

    return loss_per_sim_region


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
    """
    For internal use. Used for checking that the number of cyclones simulated is in line with model assumptions. 
    """
    cyclones_df  = pd.read_csv(
        f"{dir}/{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}_cyclone_counts.csv"
    )

    cyclone_count_per_sim = np.array(cyclones_df.filter(like = "Number of cyclones").sum(1))

    return cyclone_count_per_sim


def plot_return_period_curves(
    losses_per_sim: list, 
    legend_labels: dict, 
    figsize = (9,5),
    legend = True,
    fontsize = 12,
    n_sim=1_000_000, 
    rp=250,
    savefig = False,
):

    exceed_freq = np.linspace(1 / n_sim, 1, n_sim)
    return_period = 1 / exceed_freq[::-1]
    max_idx = int((1 - 1 / rp) * n_sim + 1)

    fig, ax = plt.subplots(figsize=figsize)

    for loss_per_sim in losses_per_sim:
        sorted_values = np.sort(loss_per_sim)

        for key, value in legend_labels.items():
            if np.array_equal(loss_per_sim, value):
                label = key

        ax.plot(return_period[:max_idx], sorted_values[:max_idx], label=label)

    ymax = ax.get_ylim()[1]
    if legend:
        ax.legend(loc = "center left", fontsize = fontsize, bbox_to_anchor=(1,0.5), ncol=1)
        
    ax.yaxis.get_offset_text().set_fontsize(fontsize)
    plt.xlabel("Return period", fontsize = fontsize)
    plt.ylabel("Loss (USD)", fontsize = fontsize)
    plt.xlim(0,250)
    plt.ylim(0, ymax)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    # plt.rcParams['font.family'] = 'Arial'
    plt.tight_layout()

    if savefig:
        plt.savefig(f"Figures/return_period_curve.png", dpi = 300, bbox_inches="tight")

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
            mean,
            sd,
            VaR_50,
            VaR_75,
            VaR_95,
            VaR_99,
            VaR_995,
            TVaR_99,
            TVaR_995,
        ]

        for key, value in column_labels.items():
            if np.array_equal(loss_per_sim, value):
                colname = key

        df[colname] = all_measures

    df_styled = df.style.format(lambda x: f"{x/units:,.0f}")

    return df_styled
