'''
Results
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp

def get_exceedance_probability_curve(loss_df, plot = True):
    # Jovana sim_utils
    loss_per_sim = loss_df.groupby('simulation')['total_loss'].sum()
    sorted_values = np.sort(loss_per_sim)

    # Calculate exceedance probabilities
    n = sorted_values.size
    exceedance_prob = np.sort(np.arange(1,n+1)/n * 100)[::-1]
    
    if plot:
        # Create a frequency-exceedance plot
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_values, exceedance_prob, marker='.', linestyle='-')
        plt.xlabel("Loss")
        plt.ylabel("Exceedance Probability (%)")
        plt.title("Aggregate Exceedance Probability Curve")
        plt.grid()
        plt.show()

    return exceedance_prob, sorted_values


def get_return_period_curve(loss_df, plot = True, rp=250):
    loss_per_sim = loss_df.groupby('simulation')['total_loss'].sum()
    sorted_values = np.sort(loss_per_sim)

    # Calculate Return Periods
    n = sorted_values.size
    exceed_freq = np.linspace(1/n, 1, n)
    return_period = 1/exceed_freq[::-1]
    max_idx = int((1-1/rp)*n+1)

    if plot:
        # Create a return-period plot
        plt.figure(figsize=(7,4))
        plt.plot(return_period[:max_idx], sorted_values[:max_idx])
        plt.xlabel("Return Period (years)")
        plt.ylabel("Loss (USD)")
        plt.title("Aggregate Exceedance Frequency Curve")
        plt.grid(color = 'whitesmoke')
        plt.show()

    return return_period, sorted_values


def risk_measures(loss_df, printing = True):
    losses = loss_df.groupby('simulation')['total_loss'].sum()

    quantiles = [25,50,75,95,99,99.5]

    quantile_25 = np.percentile(losses, 25)
    quantile_50 = np.percentile(losses, 50)
    quantile_75 = np.percentile(losses, 75)
    quantile_95 = np.percentile(losses, 95)
    quantile_99 = np.percentile(losses, 99)
    quantile_995 = np.percentile(losses, 99.5)
    mean = losses.mean()
    sd = losses.std()
    TVaR_99 = np.mean(losses[losses > quantile_99])
    TVaR_995 = np.mean(losses[losses > quantile_995])

    all_measures = [quantile_25, quantile_50, quantile_75, quantile_95, quantile_99, quantile_995, mean, sd, TVaR_99, TVaR_995]

    if printing:
        print(f'Mean Losses: {all_measures[6]:,.0f}\nStandard Deviation Losses: {all_measures[7]:,.0f} \n25% Quantile: {all_measures[0]:,.0f} \n50% Quantile: {all_measures[1]:,.0f} \n75% Quantile: {all_measures[2]:,.0f} \n95% Quantile: {all_measures[3]:,.0f} \n99% Quantile: {all_measures[4]:,.0f} \n99.5% Quantile: {all_measures[5]:,.0f} \n1% TVaR: {all_measures[-2]:,.0f} \n0.5% TVaR: {all_measures[-1]:,.0f}\n')

    return all_measures