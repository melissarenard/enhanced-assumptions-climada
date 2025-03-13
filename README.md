# Stochastic variability in natural catastrophe models: application to tropical cyclones in Australia
Author: Melissa Renard

This folder holds all files necessary to generate catastrophe losses under numerous sets of assumptions. We use the open-source CAT model CLImate ADAptation (CLIMADA) to estimate losses in Australia from tropical cyclones (TCs). CLIMADA makes certain assumptions about natural hazards and associated losses, which are similar to the assumptions that other non-proprietary CAT models make. Here we introduce enhanced probabilistic assumptions that apply to TCs:

1. A Markov-modulated non-homogeneous Poisson process for hazard occurrence. The process is modulated by a discrete-time Markov chain for El Niño-Southern Oscillation (ENSO). The non-homogeneous part specifically allows for seasonality;
2. Differences in TC genesis location based on ENSO (simulated by the Markov chain);
3. Differences in TC intensity based on ENSO (simulated by the Markov chain);
4. Greater losses from consecutive events in a given area.

## Files
### Data

* `ENSO.csv`: Data of ENSO phase attributable to each year.

### Setting up assumptions

1. `Historical tracks df.ipynb`: Complies TC data into `tracks.csv` in `Data` folder.
2. `tracks_r_tables.Rmd`: Creates files in `R tables` folder.
3. `Hazard generation.ipynb`: Synthetic hazard generation for this project; saves it as `haz_aus_300synth.hdf5` in `Hazards` folder. **If `haz_aus_300synth.hdf5` does not appear in the `Hazards` folder then you must run this code first before moving on to the next part.**

### Running simulations

* `CLIMADA_loss_dfs_AUS.ipynb`: notebook that runs the functions in `sim_mel.py`. **Use this notebook to run simulations and save output to `Output` folder.**
* `sim_mel.py`: Python script with new CLIMADA assumptions.
* `results.py`: Python script with functions to show or plot results of interest.

### TC data analysis (internal use only)

* `TC_data_analysis.Rmd`: R markdown file that conducts data analysis on TCs in the South Indian (SI) and South Pacific (SP) basins. The data analysis helped in setting the assumptions, but the file itself does not directly feed into any part of the loss output generation.
* `haz_check.ipynb`: Python notebook that analysis CLIMADA's Hazard classes `TCTracks` and `TropCyclone` and their functions within.

## CLIMADA
**You must first install CLIMADA before using this code (CLIMADA petals not necessary). Instructions on installation are [here](https://climada-python.readthedocs.io/en/stable/guide/install.html).**
Documentation on CLIMADA can be found [here](https://climada-python.readthedocs.io/en/stable/index.html). 
