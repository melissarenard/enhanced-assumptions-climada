"""
Simulation for CLIMADA
"""
from climada.hazard import TropCyclone
from climada.engine import ImpactCalc
import pandas as pd
import numpy as np
import datetime as dt
import scipy.sparse as sp
import scipy.integrate as integrate
import scipy.optimize as optimize
from collections import Counter
from bisect import bisect_right
from joblib import Parallel, delayed

enso_df = pd.read_csv('data/ENSO.csv', skiprows=3) # years and corrresponding ENSO phases
enso_df = enso_df._append({'year': 2024, 'enso': 'Nino'}, ignore_index=True)

# ENSO Markov Chain transition matrix.
# States (in row/column order): La Nina, neutral, El Nino.
enso_MC = pd.read_csv('R tables/enso_MC.csv', index_col = 0).values

n_tracks_CLIMADA = pd.read_csv('R tables/n_tracks_CLIMADA.csv')
par_MMNHPP = pd.read_csv('R tables/MMNHPP_par.csv') # MMNHPP hyperparameters

# Northern and Southern Hemispheres
my_keys_SH = [dt.datetime(year-1, 8, 1) for year in enso_df['year']]
my_keys_NH = [dt.datetime(year, 1, 1) for year in enso_df['year']]

enso_dict_SH = {my_keys_SH[i]: enso_df['enso'].values[i] for i in range(enso_df.shape[0])}

enso_dict_NH = {my_keys_NH[i]: enso_df['enso'].values[i] for i in range(enso_df.shape[0])}

basin_to_enso_dict = {'EP': enso_dict_NH, 'NA': enso_dict_NH, 'NI': enso_dict_NH, 'SI': enso_dict_SH, 'SP': enso_dict_SH, 'WP': enso_dict_NH}

# Translating ENSO phase to integer and vice-versa
phase_to_num = {'Nina': -1, 'Neutral': 0, 'Nino': 1}
num_to_phase = {value: key for key, value in phase_to_num.items()}

# Average maximum TC intensity by ENSO phase, split by basin
intensity_df = pd.read_csv('R tables/severity_enso.csv')

###### 
# Note: year for Southern Hemisphere starts on 1 August. Only looking at tracks in Southern Hemisphere (SI and SP basins).

def date_to_enso(date, enso_dict):
    '''
    Returns the ENSO phase corresponding to a date. ENSO phase is slightly delayed in southern hemisphere relative to northern hemisphere to be consistent with tropical cyclone seasons.

    Parameters
    ----------

    date : datetime datetime. Must be between 1 January 1896 and 31 December 2023 for northern hemisphere or between 1 August 1895 and 31 July 2023 for southern hemisphere.

    enso_dict : dictionary
        Differentiates between Northern and Southern Hemisphere. Can take the value of either enso_dict_NH or enso_dict_SH

    Returns: str
        Either 'Nina', 'Neutral' or 'Nino'.

    '''
    # ChatGPT: bisect_right
    # dict_keys_sorted = np.sort([*enso_dict.keys()])
    dict_keys_sorted = list(enso_dict.keys()) # sort keys of enso_dict
    index = bisect_right(dict_keys_sorted, date) # Return the index where to insert item x in list a, assuming a is sorted.
    last_key = dict_keys_sorted[-1] 
    if date >= dt.datetime(last_key.year+1, last_key.month, last_key.day) or index == 0: # Date out of range of the enso_dict
        raise ValueError(f'Date {date} out of range.') 
    else:
        return enso_dict[dict_keys_sorted[index-1]] 

def get_enso(haz):
    '''
    Returns the ENSO phase of each hazard in Hazard object, as a list in the order of the Hazard object.
    
    Parameters
    ----------
    haz : TropCyclone object
    
    Returns
    -------
    haz_enso : numpy array
        numpy array of the same length as the size of haz, with each element corresponding to the ENSO phase of each hazard in haz.
    '''
    haz_enso = np.empty(haz.size, dtype = object) # empty array

    basins = haz.basin # array of basin in which each event formed
    dates = haz.date # array of first recorded time of each event

    for i in range(haz.size):
        basin = basins[i]
        date = dt.datetime.fromordinal(dates[i])
        haz_enso[i] = date_to_enso(date, basin_to_enso_dict[basin]) # from date of event to corresponding ENSO phase
    
    return haz_enso # array of ENSO phase in which each event formed (according to  the first recorded time)


def sim_enso(n_years, starting_enso='Nino', matrix=enso_MC): 
    '''
    Simulates ENSO phases as a Markov Chain from a given probability transition matrix. Time interval is 1 year.

    Parameters
    ----------
    starting_enso : string, ENSO phase
        ENSO phase corresponding to the most recent year. Default : 'Nino' (corresponding to the phase of 2023).
    
    n_years : int
        Number of years to simulate the Markov Chain.

    matrix : mumpy array
        3 x 3 matrix with transition probabilities from one ENSO phase to another. Rows must sum to 1.

    Returns
    -------
    phases : list
        List of length n_years of the simulated ENSO phases.
    '''
    if n_years < 1: # Must simulate at least 1 year
        raise ValueError(f"n_years = {n_years} is invalid, must be a positive integer.")
    phases = np.empty(n_years, dtype = object) # empty array of length n_years
    prev_phase = phase_to_num[starting_enso] # integer corresponding to starting_enso
    
    for i in range(n_years):
        new_phase = np.random.choice([-1,0,1], p=matrix[prev_phase+1]) # randomly select next ENSO phase based on the ENSO Markov chain transition matrix
        phases[i] = num_to_phase[new_phase] # ENSO phase corresponding to new integer
        prev_phase = new_phase # replace value of prev_phase for the next year of the Markov chain

    return phases # array of each simulated ENSO phase


def get_MMNHPP_param(basin, enso, par_df = par_MMNHPP): 
    '''
    Extract estimated Least squares parameters for the NHPP corresponding to a basin and ENSO phase
    '''
    return par_df.loc[par_df['enso'] == enso][basin].values # The 4 hyperparameters of the MMNHPP for a given ENSO phase and basin


def MMNHPP_seasonal(t, basin, enso, par_df = par_MMNHPP): 
    '''
    Returns the value of the seasonal function as a function of t (time), for a given ENSO phase and basin.
    '''
    coefs = get_MMNHPP_param(basin = basin, enso = enso, par_df = par_df)

    return coefs[0] + coefs[1]*np.exp(coefs[2]*np.sin(2*np.pi*t + coefs[3]))


def MMNHPP_lambda(t, basin, enso, par_df = par_MMNHPP):   
    '''
    Return the value of the esimated lambda as a function of t (positive version of MMNHPP_seasonal) for the NHPP corresponding to a basin and ENSO phase
    '''
    if basin in ['SI', 'SP']:
        return np.maximum(MMNHPP_seasonal(t+7/12, basin, enso, par_df), 0) # season starts in August
    elif basin in ['EP', 'NA', 'NI', 'WP']:
        return np.maximum(MMNHPP_seasonal(t, basin, enso, par_df), 0)
    else: 
        raise ValueError(f"basin = {basin} is invalid, must be 'EP', 'NA', 'NI', 'SI', 'SP', or 'WP'.")

bounds_MMNHPP_df = pd.DataFrame(columns = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP'])
bounds_MMNHPP_df = bounds_MMNHPP_df.reindex(range(3))
for basin in ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']:
    for enso in ['Nina', 'Neutral', 'Nino']:
        bound = -optimize.minimize_scalar(lambda t: -MMNHPP_lambda(t, basin, enso, par_df=par_MMNHPP), bounds = [0,1], method='bounded')['fun'] # maximum value of MMNHPP_lambda

        bounds_MMNHPP_df[basin][phase_to_num[enso]+1] = bound # attribute the bound to cell in Pandas dataframe.


def sample_MMNHPP_lambda(basin, enso, bounds_df = bounds_MMNHPP_df, par_df = par_MMNHPP):
    '''
    Sample a value from the distribution of MMNHPP_lambda. Sampled value corresonds to an event time. Sampling algorithm is an acceptance-rejection algorithm.
    '''
    # https://stackoverflow.com/questions/66874819/random-numbers-with-user-defined-continuous-probability-distribution
    bound = bounds_df[basin][phase_to_num[enso]+1] # maximum y value

    while True: 
        x = np.random.uniform(0,1) # uniform between the bounds of time
        y = np.random.uniform(0, bound) # uniform between the bounds of the pdf

        if y < MMNHPP_lambda(x, basin, enso, par_df): # accept if (x,y) falls under the curve
            return x 


def sample_MMNHPP_lambda_n(n_events, basin, enso, bounds_df = bounds_MMNHPP_df, par_df = par_MMNHPP):
    '''
    Sample n values from the distribution of MMNHPP_lambda. Sampled values correspond to event times. Sampling algorithm is an acceptance-rejection algorithm.
    '''
    event_times = np.empty(n_events, dtype=object) # empty array
    for i in range(n_events):
        event_times[i] = sample_MMNHPP_lambda(basin, enso, bounds_df, par_df) # fill out array

    return event_times


def yearrange(haz):
    '''
    Get year range of Hazard object. The year of the first event and the year of the last event.
    
    Parameters
    ----------
    haz : Hazard object
    
    Returns
    -------
    yearrange : tuple of size 2
        the first and last year of hazards observed in haz.
    
    '''

    yearrange = (dt.datetime.fromordinal(np.min(haz.date)).year, # first event year
                 dt.datetime.fromordinal(np.max(haz.date)).year) # last event year
    
    return yearrange


def delta_years(haz):
    '''
    Return the total number of years that a Hazard object spans.
    '''
    delta_years = max(yearrange(haz)) - min(yearrange(haz)) + 1
    
    return delta_years

### Simulate intensity function

def sim_intensity(haz, enso): 
    '''
    Modify intensity of hazards in Hazard object based on the ENSO phase. The hazards in the Hazard object are all in the same basin.
    '''
    basin = np.unique(haz.basin).item() # if there is more than one basin, then the function will not work.
    new_haz = TropCyclone() # new_haz is an empty Hazard object
    new_haz.append(haz) # new_haz is a shallow copy of haz
    for i in range(haz.size):
        my_enso = get_enso(haz)[i] # ENSO phase of each hazard
        new_haz.intensity[i] = haz.intensity[i] * intensity_df[basin][phase_to_num[enso]+1] / intensity_df[basin][phase_to_num[my_enso]+1] # multiply by the ratio of average intensity in chosen enso over average intensity in ENSO phase of the actual event.

    return new_haz

def enso_probs(enso_sim, p):
    '''
    Return an array of probabilities summing to 1, corresponding to the probability of selecting an event from the event set from La Niña, Neutral and El Niño, respectively.
    '''
    if enso_sim not in ['Nina', 'Neutral', 'Nino']:
        raise ValueError("Invalid value for enso_sim. Must be 'Nina', 'Neutral' or 'Nino'.")
    
    q = (1-p)/2 # probabilities sum to 1.

    if enso_sim == 'Nina':
        probs = np.array([p, q, q])
    elif enso_sim == 'Neutral':
        probs = np.array([q, p, q])
    elif enso_sim == 'Nino': 
        probs = np.array([q, q, p])

    return probs

def select_hazards(haz, params, haz_dicts, loc, p_loc, intensity):
    '''
    Select hazards in a basin based on the number of occurrences in the basin.
    Input haz_sim and event_times, and output those.
    '''
    enso_sim, basin, n_events, haz_sim = params # expand params

    event_ids_basin_dict, haz_basin_dict, enso_basin_dict = haz_dicts # expand haz_dicts
    
    event_ids_basin = np.setdiff1d(event_ids_basin_dict[basin], haz_sim.event_id) # Return the unique values in `ar1` that are not in `ar2`.

    if loc:
        haz_basin = haz_basin_dict[basin] # filter haz by those in the basin
        enso_basin = enso_basin_dict[basin] # ENSO phase for each track in haz_basin
        result = Counter(np.random.choice(['Nina', 'Neutral', 'Nino'], size=n_events, p = enso_probs(enso_sim, p_loc))) # randomly choose from which ENSO phase we will obtain tracks for each event occurrence

        event_ids_sim = np.array([], dtype = int)

        for enso, count in result.items():
            event_ids_basin_enso = np.setdiff1d(haz_basin.event_id[enso_basin == enso], haz_sim.event_id) # event ids in haz in respective basin and enso phase
            if len(event_ids_basin_enso) > 0:
                event_ids_sim = np.append(event_ids_sim, np.random.choice(event_ids_basin_enso, count, replace = False))

    else: 
        event_ids_sim = np.random.choice(event_ids_basin, n_events, replace = False)

    haz_sim_basin = haz.select(event_id = event_ids_sim)

    if intensity:
        haz_sim_basin = sim_intensity(haz_sim_basin, enso_sim)
    
    return haz_sim_basin


########## Simulate Hazards ##########

def sim_haz_orig(haz, n_years = 1): 
    '''
    Simulate tropical cyclones according to a Poisson distribution. Sampling of the hazard events is done without replacement.

    Parameters
    ----------

    haz : TropCyclone object

    n_years : int
        number of years of tropical cyclones to generate. Default: 1

    Returns
    -------
    haz_sim : TropCyclone object

    '''
    # Jovana
    ev_per_year = haz.calc_year_set()
    start_year, end_year = yearrange(haz)
    for year in range(start_year, end_year + 1):
        if year not in ev_per_year.keys():
            ev_per_year[year] = np.array([])
    tc_per_year = [ev_per_year[i].size for i in ev_per_year.keys()]

    # get average events per year (lambda)
    lam = np.mean(tc_per_year)

    # simulate Poisson (lambda) events
    number_of_events = np.random.poisson(lam*n_years)
    if number_of_events > 0:
        # randomly sample number_of_events from hazard event set
        event_ids_sim = np.random.choice(haz.event_id, number_of_events, replace = False)
        haz_sim = haz.select(event_id = event_ids_sim)
    else: 
        haz_sim = TropCyclone()
    
    return haz_sim


def sim_haz_MMNHPP(haz, haz_info, haz_dicts, n_years = 1, starting_enso = 'Nino', loc = False, p_loc = 0.5, intensity = False): 
    '''
    Simulate number of events and event times based on a Markov-modulated non-homogeneous Poisson Process. 
    
    Parameters
    ----------
    haz : Hazard object
        must be only tracks occurring between 1980 and 2023 (for now).

    haz_info : 
    
    haz_dicts : 

    n_years : number of years of tropical cyclones to generate. Default: 1.
    
    starting_enso : string
        starting ENSO phase for simulating the next phase in the ENSO Markov Chain.
        Possible values are 'Nina', 'Neutral' and 'Nino'. Default: 'Nino'.

    loc : bool
        If True, select hazards based on geographical location as a function of the simulated ENSO phase. Default: False.

    intensity : bool
        If True, adjust intensity of tracks at each time point as a function of the simulated ENSO phase. Default: False.
        
    Returns
    -------
    haz_sim : TropCyclone object
        Contains the simulated events.
    
    event_times : numpy array
        Contains the times of each corresponding simulated event in haz_sim. Times are all positive and in units of years. 

    '''
    # https://en.wikipedia.org/wiki/Poisson_point_process#Simulation
    # Conditional on the number of events: standardised pdf is 
    # lambda(t) / Lambda(T).

    enso_sims = sim_enso(n_years, starting_enso)

    basins, haz_hist_basins, Lambda_MMNHPP_df = haz_info

    event_times = []
    haz_sim = TropCyclone()

    for i in range(n_years):
        enso_sim = enso_sims[i]
        for basin in basins:
            lam = Lambda_MMNHPP_df[basin][phase_to_num[enso_sim]+1] # inhomogeneous

            n_events = np.random.poisson(lam)

            if n_events > 0:
                event_times_basin = sample_MMNHPP_lambda_n(n_events, basin, enso_sim) + i

                params = (enso_sim, basin, n_events, haz_sim)

                haz_sim_basin = select_hazards(haz, params, haz_dicts, loc, p_loc, intensity) # Selecting hazards based on 'loc' and adjusting intensity based on 'intensity'
                haz_sim.append(haz_sim_basin) # add hazards in this basin to hazard output
                event_times.extend(event_times_basin) # add event times in this basin to event times output

    return haz_sim, np.array(event_times)


########## Simulate Losses ##########

### Damage functions
def decay_fn(t): 
    if 0 <= t < 1:
        return np.exp(-5*t)
    elif t < 0:
        raise ValueError(f't = {t} invalid, t must be positive')
    else:
        return 0


def sim_damage(event_times, loss_df): 
    '''
    Adjusts the value of damage if two consecutive events occur in the same location, according to the function:

    theta(tau) = e^(-5tau), 0 <= tau < 1
    theta(tau) = 0 otherwise

    Parameters
    ----------

    event_times : array
        Event times.

    loss_df : sparse matrix
        Each row corresponds to an event and each column corresponds to a grid cell of exposure. 

    Returns
    -------

    loss_df : sparse matrix
        Each row corresponds to an event and each column corresponds to a grid cell of exposure. The returned loss_df losses are weakly greater than what was given as an input.

    '''
    # ChatGPT help for the below 3 lines:
    sorted_indices = np.argsort(event_times) # index values of sorted list
    event_times = event_times[sorted_indices] # sort event times
    loss_df = loss_df[sorted_indices] # sort df according to event times

    interarrival_times = np.diff(event_times) # interarrival times

    for i in range(interarrival_times.size):
        current_row = loss_df[i].toarray().flatten() > 0
        next_row = loss_df[i+1].toarray().flatten() > 0

        if np.any(current_row & next_row):
            loss_df[i+1, current_row & next_row] *= (1+decay_fn(interarrival_times[i]))

    return loss_df


### Loss multi-year

def sim_losses_n_year(haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years = 1, starting_enso = 'Nino', loc = False, p_loc = 0.5, intensity = False, damage = False):
    '''
    Returns one simulation of losses from a number of years of tropical cyclones.

    Parameters
    ----------
    haz : TropCyclone object

    haz_info : 

    haz_dicts : 

    exp : Exposure

    impfset : Impact

    fqcy : string
        Frequency/process choice for simulating the losses.
        Possible values are 'orig' (CLIMADA assumption, Poisson random variable) and 'MMNHPP'.

    n_years : int
        number of years of TCs for a single simulation. Default: 1

    starting_enso : string
        starting ENSO phase for simulating the next phase in the ENSO Markov Chain.
        Possible values are 'Nina', 'Neutral' and 'Nino'. Default: 'Nino'.

    loc : bool
        if True, select tracks from haz based on simulated ENSO phase. Default: False

    intensity : bool
        if True, adjust intensity of tracks at each time point as a function of the simulated ENSO phase. Only relevant for 'MMNHPP'. Default: False

    damage : bool
        If True, increase the damage of consecutive events using the function 'sim_damage'. Default: False

    Returns
    -------
    
    loss_df : sparse matrix
        Matrix of monetary losses with rows corresponding to a tropical cyclone occurring in the period, and columns corresponding to a grid cell of the exposure.
    '''
    if fqcy == 'orig':
        haz_sim = sim_haz_orig(haz, n_years)
    elif fqcy == 'MMNHPP':
        sim = sim_haz_MMNHPP(haz, haz_info, haz_dicts, n_years, starting_enso, loc, p_loc, intensity)
        haz_sim = sim[0]
        event_times = sim[1]
    else:
        raise ValueError(f"fqcy = '{fqcy}' is invalid, must be 'orig' or 'MMNHPP'.")
    
    # With the help of Jovana's code in sim_utils
    if haz_sim.size > 0:
        loss_df = ImpactCalc(exp, impfset, haz_sim).impact(assign_centroids=False).imp_mat # calc impact

        if damage:
            loss_df = sim_damage(event_times, loss_df)

    else: 
        loss_df = sp.csr_matrix((0, exp.gdf.shape[0]))

    return loss_df

def sim_single_loss(haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years = 1, starting_enso = 'Nino', loc = False, p_loc = 0.5, intensity = False, damage = False):
    
    loss_df = sim_losses_n_year(haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years, starting_enso, loc, p_loc, intensity, damage)

    result = sp.csr_matrix(loss_df.sum(0))

    return result

def run_simulation(seed, haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years, starting_enso, loc, p_loc, intensity, damage, n_exp):
    # Set the seed for reproducibility within this process
    np.random.seed(seed)
    
    result = sim_single_loss(haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years, starting_enso, loc, p_loc, intensity, damage)
    
    # Ensure the result has the correct shape
    if result.shape[1] != n_exp:
        raise ValueError(f"Incompatible dimensions: result.shape[1] = {result.shape[1]}, expected {n_exp}")
    
    return result

    
def sim_losses_per_point(haz, exp, impfset, fqcy, n_sim, n_years, starting_enso='Nino', loc=False, p_loc=0.5, intensity=False, damage=False, seed=23, num_cpus=None):
    '''
    Returns simulations of losses from a number of years of tropical cyclones.
    
    Parameters
    ----------
    haz : TropCyclone object
    
    exp : Exposure object
    
    impfset : Impact
    
    fqcy : string
        Frequency/process choice for simulating the losses.
        Possible values are 'orig' and 'MMNHPP'.
    
    n_sim : int
        Number of simulations to generate.
    
    n_years : int
        Number of years for a single simulation.
    
    starting_enso : string
        Starting ENSO phase for simulating the next phase in the ENSO Markov Chain. Possible values are 'Nina', 'Neutral', and 'Nino'. Default: 'Nino'.
    
    loc : bool
        If True, select tracks from haz based on simulated ENSO phase. Default: False

    p_loc : float
        Value between 0 and 1
    
    intensity : bool
        If True, adjust intensity of tracks at each time point as a function of the simulated ENSO phase. Only relevant for 'MMNHPP'. Default: False
    
    damage : bool
        If True, increase the damage of consecutive events using the function 'sim_damage'. Default: False
    
    seed : int
        Random seed for simulations. Default: 23
    
    Returns
    -------
    sim_losses_per_pt : sparse matrix
        Matrix of monetary losses where the rows correspond to a simulation and the columns correspond to a grid cell of the exposure.
    '''
    
    if fqcy == 'orig':
        loc = False
        intensity = False
        damage = False
        print("If fqcy = 'orig' then loc, intensity, and damage are not used. They are all set to False.")
    
    # Set the main seed
    np.random.seed(seed)
    
    # Generate a list of seeds for each simulation
    seeds = np.random.randint(0, 2**31 - 1, size=n_sim)
    
    # Setup
    n_exp = exp.gdf.shape[0]
    haz_enso = get_enso(haz)
    haz_basins = np.array(haz.basin)
    basins = np.sort(np.unique(haz.basin))
    haz_hist = haz.select(orig=True)
    haz_hist_basins = np.array(haz_hist.basin)
    Lambda_MMNHPP_df = pd.DataFrame(columns=basins).reindex(range(3))
    
    event_ids_basin_dict = {}
    haz_basin_dict = {}
    enso_basin_dict = {}
    for basin in basins:
        event_ids_basin_dict[basin] = haz.event_id[haz_basins == basin]
        haz_basin_dict[basin] = haz.select(event_id=event_ids_basin_dict[basin])
        enso_basin_dict[basin] = haz_enso[haz_basins == basin]
    
        n_hist_tracks = np.sum(haz_hist_basins == basin)
    
        for enso_sim in ['Nina', 'Neutral', 'Nino']:
            lam = integrate.quad(lambda t: MMNHPP_lambda(t, basin, enso_sim), 0, 1)[0] * n_hist_tracks / n_tracks_CLIMADA[basin][0]  # inhomogeneous
            Lambda_MMNHPP_df[basin][phase_to_num[enso_sim] + 1] = lam
    
    haz_info = (basins, haz_hist_basins, Lambda_MMNHPP_df)
    haz_dicts = (event_ids_basin_dict, haz_basin_dict, enso_basin_dict)
    
    # Parallel processing with joblib
    results = Parallel(n_jobs=num_cpus or -1)(
        delayed(run_simulation)(
            seeds[idx], haz, haz_info, haz_dicts, exp, impfset, fqcy, n_years, starting_enso, loc, p_loc, intensity, damage, n_exp
        ) for idx in range(n_sim)
    )
    
    # Combine results into a single sparse matrix
    sim_losses_per_pt = sp.vstack(results)
    
    return sim_losses_per_pt
