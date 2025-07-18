# Stochastic variability in natural catastrophe models: application to tropical cyclones in Australia
Authors: Patrick Laub, Melissa Renard, Bernard Wong

This folder holds all files necessary to generate catastrophe losses under numerous sets of assumptions. We use the open-source CAT model CLImate ADAptation (CLIMADA) to estimate losses in Australia from tropical cyclones (TCs). CLIMADA makes certain assumptions about natural hazards and associated losses, which are similar to the assumptions that other non-proprietary CAT models make. Here we introduce enhanced probabilistic assumptions that apply to TCs:

1. A Markov-modulated non-homogeneous Poisson process for hazard occurrence. The process is modulated by a discrete-time Markov chain for El Niño-Southern Oscillation (ENSO). The non-homogeneous part specifically allows for seasonality;
2. Differences in TC genesis location based on ENSO (simulated by the Markov chain);
3. Differences in TC intensity based on ENSO (simulated by the Markov chain);
4. Greater losses from consecutive events in a given area.

## Instructions
1. Run `Hazard generation.ipynb`
2. Run `Generate Year Loss Tables.ipynb`
3. Optional: Use `Analayse YLTs.ipynb` to import `natcat_analysis.py` and call relevant functions to obtain summary statistics/plots of results.

* After Step 1, the hazard file will be added to the `Hazards` folder.
* After Step 2, loss output files will be added to the `Outputs` folder. 

## Files
### Data

* `ENSO.csv`: Data of ENSO phase attributable to each year.
* `tracks.csv`: Tracks loaded into CLIMADA are saved in this csv with variables of interest for data analysis in R.

### Setting up assumptions

1. `Historical tracks df.ipynb`: Complies TC data into `tracks.csv` in `Data` folder.
2. `tracks_r_tables.Rmd`: Creates files in `R tables` folder.
3. `Hazard generation.ipynb`: Synthetic hazard generation for this project; saves it as `haz_aus_300synth.hdf5` in `Hazards` folder.

### Running simulations

* `Generate Year Loss Tables.ipynb`: notebook that runs the functions in `adjusted_year_loss_tables.py`.
* `adjusted_year_loss_tables.py`: Python script with new CLIMADA assumptions.
* `Analyse YLTs.ipnb`: notebook that generates plots and tables from the previously generated YLTs.
* `natcat_analysis.py`: Python script with functions to show or plot results of interest.

### TC data analysis (internal use only)

* `TC_data_analysis.Rmd`: R markdown file that conducts data analysis on TCs in the South Indian (SI) and South Pacific (SP) basins. The data analysis helped in setting the assumptions, but the file itself does not directly feed into any part of the loss output generation.
* `haz_check.ipynb`: Python notebook that analyses CLIMADA's Hazard classes `TCTracks` and `TropCyclone` and their functions within.
* `TCSurge.ipynb`: Python notebook that analyses CLIMADA's "petals" class `TCSurgeBathtub` for modelling storm surge.

## CLIMADA

**You must first install CLIMADA before using this code (CLIMADA petals not necessary). Instructions on installation are [here](https://climada-python.readthedocs.io/en/stable/guide/install.html).**
Documentation on CLIMADA can be found [here](https://climada-python.readthedocs.io/en/stable/index.html). 

After installing, you will also need to manually download the dataset [gpw_v4_population_count_rev11_2020_30_sec.tif](http://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/gpw-v4-population-count-rev11/gpw-v4-population-count-rev11_2020_30_sec_tif.zip) and place it in `~/climada/data`.
A free NASA Earthdata login is required.
CLIMADA will download `IBTrACS.ALL.v04r01.nc` to the same directory when the hazard generation code is first run (we take note of the dataset version here for the purpose of reproducibility).
