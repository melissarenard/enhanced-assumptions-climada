# Standard library
import os
import datetime as dt
from typing import Optional
from time import time
from collections import Counter
from bisect import bisect_right

# Third-party libraries
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
import geopandas as gpd

from numba import njit
from scipy import integrate
from scipy import optimize
from tqdm import trange, tqdm

# Domain-specific libraries
from climada.hazard import TropCyclone
from climada.entity.exposures import LitPop
from climada.engine import ImpactCalc
from climada.entity.impact_funcs import ImpactFuncSet

ALL_BASINS = ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']
BASIN_TO_IDX = {b: i for i, b in enumerate(ALL_BASINS)}

ENSO_PHASES = ['Nina', 'Neutral', 'Nino']
ENSO_TO_IDX = {p: i for i, p in enumerate(ENSO_PHASES)}


def date_to_enso(date: dt.datetime, enso_dict: dict[dt.datetime, str]) -> str:
    """
    Given a date, return the corresponding ENSO phase (Nina, Neutral, or Nino)
    based on the most recent ENSO phase start date in the given dictionary.

    Parameters
    ----------
    date
        Date to lookup. Should fall within the range of enso_dict keys.
    enso_dict
        Maps datetime keys (phase start dates) to ENSO phase strings.

    Returns
    -------
    str
        ENSO phase for the given date.

    Raises
    ------
    ValueError if the date is outside the range of known ENSO periods.
    """
    keys_sorted = sorted(enso_dict.keys())
    index = bisect_right(keys_sorted, date)

    if index == 0 or date >= keys_sorted[-1] + dt.timedelta(days=366):
        raise ValueError(f"Date {date} is out of range of ENSO phase definitions.")

    return enso_dict[keys_sorted[index - 1]]


def enso_phases_of_historical_tracks(haz: TropCyclone) -> npt.NDArray[np.str_]:
    """
    Get ENSO phase (Nina, Neutral, Nino) for each cyclone event based on
    the date of its first record and basin.

    Parameters
    ----------
    haz
        The hazard object containing cyclone metadata.

    Returns
    -------
    npt.NDArray[object]
        Array of ENSO phases (as strings) for each event in `haz`.
    """
    enso_df = pd.read_csv('Data/ENSO.csv', skiprows=3)
    enso_df = enso_df._append({'year': 2024, 'enso': 'Nino'}, ignore_index=True)

    # Build ENSO phase start dates for NH and SH
    enso_dict_NH = {
        dt.datetime(year, 1, 1): phase
        for year, phase in zip(enso_df['year'], enso_df['enso'])
    }
    enso_dict_SH = {
        dt.datetime(year - 1, 8, 1): phase
        for year, phase in zip(enso_df['year'], enso_df['enso'])
    }

    # Which dictionary to use for each basin
    basin_to_dict = {
        'EP': enso_dict_NH, 'NA': enso_dict_NH, 'NI': enso_dict_NH, 'WP': enso_dict_NH,
        'SI': enso_dict_SH, 'SP': enso_dict_SH
    }

    haz_enso = np.empty(haz.size, dtype=object)

    for i in range(haz.size):
        basin = haz.basin[i]
        date = dt.datetime.fromordinal(haz.date[i])
        enso_dict = basin_to_dict[basin]
        haz_enso[i] = date_to_enso(date, enso_dict)

    return haz_enso


def get_intensity_adjustment_factor(
    basin: str,
    enso_actual: str,
    enso_simulated: str,
    max_intensity: npt.NDArray,
) -> float:
    """
    Return the multiplicative adjustment factor to apply to cyclone intensity
    when simulating it under a different ENSO phase.

    Parameters
    ----------
    basin
        The basin in which the cyclone occurred.
    enso_actual
        The ENSO phase during which the cyclone originally occurred.
    enso_simulated
        The ENSO phase being simulated.
    max_intensity
        Maximum intensity of cyclones in each basin, by ENSO phase.
        Shape: (3, n_basins)

    Returns
    -------
    float
        Multiplicative intensity adjustment factor.
    """
    actual_phase_num = ENSO_TO_IDX[enso_actual]
    sim_phase_num = ENSO_TO_IDX[enso_simulated]
    return max_intensity[sim_phase_num, BASIN_TO_IDX[basin]] / max_intensity[actual_phase_num, BASIN_TO_IDX[basin]]


def compute_loss_catalogue(
    haz: TropCyclone,
    exp: LitPop,
    impfset: ImpactFuncSet,
    output_dir="Outputs",
    filename_base="loss_catalogue",
    save_catalogue=False,
) -> tuple[dict[str, sp.csr_matrix], dict[int, int]]:
    """
    Compute loss catalogues for all ENSO phases, with and without intensity adjustment.

    Returns
    -------
    loss_catalogues : dict[str, sp.csr_matrix]
        Dictionary mapping ENSO phase to loss matrix.
    loss_catalogue_index : dict[int, int]
        Mapping from event_id to row index in the matrices.
    """
    os.makedirs(output_dir, exist_ok=True)

    loss_catalogues = {}
    loss_catalogue_index = {eid: i for i, eid in enumerate(haz.event_id)}

    # Average maximum TC intensity by ENSO phase, split by basin
    max_intensity_df = pd.read_csv(
        "R tables/severity_enso.csv"
    )
    assert all(max_intensity_df.columns == ALL_BASINS), "Invalid column names in severity_enso.csv"
    max_intensity = np.array(max_intensity_df.values)

    # --- No intensity adjustment ---
    print("Computing loss catalogue: No adjustment", flush=True)
    loss_no_adjust = ImpactCalc(exp, impfset, haz).impact(assign_centroids=False).imp_mat
    loss_catalogues["No adjustment"] = loss_no_adjust
    if save_catalogue:
        sp.save_npz(os.path.join(output_dir, f"{filename_base}_no_adjustment.npz"), loss_no_adjust)

    # --- ENSO-phase-specific adjustments ---
    original_enso = enso_phases_of_historical_tracks(haz)
    original_intensities = haz.intensity.copy()

    for sim_enso in ["Nina", "Neutral", "Nino"]:
        print(f"Computing loss catalogue: {sim_enso}", flush=True)

        # Scale the intensity of each cyclone based on a given simulated ENSO phase
        adjustment_factors = np.ones(haz.size)

        for i in range(haz.size):
            basin = haz.basin[i]
            actual_enso = original_enso[i]
            adjustment_factors[i] = get_intensity_adjustment_factor(basin, actual_enso, sim_enso, max_intensity)

        haz.intensity = sp.diags(adjustment_factors) @ original_intensities

        # Compute and store
        loss_mat = ImpactCalc(exp, impfset, haz).impact(assign_centroids=False).imp_mat
        loss_catalogues[sim_enso] = loss_mat
        if save_catalogue:
            sp.save_npz(os.path.join(output_dir, f"{filename_base}_{sim_enso}.npz"), loss_mat)

    haz.intensity = original_intensities

    return loss_catalogues, loss_catalogue_index


def simulate_enso_time_series(
    n_sim: int,
    n_years: int,
    starting_enso: str,
    show_progress=False
) -> npt.NDArray[np.uint8]:
    """
    Simulate ENSO phase time series for all simulations using a Markov chain.

    Parameters
    ----------
    n_sim
        Number of simulations to run.
    n_years
        Number of years per simulation.
    starting_enso
        ENSO phase to start the chain from ('Nina', 'Neutral', 'Nino').
    

    Returns
    -------
    npt.NDArray[np.unint8]
        ENSO phases per [simulation][year], as integers (0, 1, 2).
    """
    # 3x3 ENSO Markov transition matrix.
    matrix = pd.read_csv('R tables/enso_MC.csv', index_col = 0).values

    simulations = np.empty((n_sim, n_years), dtype=np.uint8)
    start_val = ENSO_TO_IDX[starting_enso]

    for sim in trange(n_sim, desc="Simulating ENSO time series", disable=not show_progress):
        prev_phase = start_val
        for year in range(n_years):
            next_phase = np.random.choice([0, 1, 2], p=matrix[prev_phase])
            simulations[sim, year] = next_phase
            prev_phase = next_phase

    return simulations


def simulate_number_of_cyclones(
    n_sim: int,
    n_years: int,
    poisson_params: npt.NDArray[np.float64],
    basins: list[str],
    enso_simulations: npt.NDArray[np.uint8],
    show_progress=False,
) -> npt.NDArray[np.uint8]:
    """
    Simulate the number of tropical cyclones per basin and year.

    Parameters
    ----------
    n_sim
        Number of simulations.
    n_years
        Number of years per simulation.
    poisson_params
        MMNHPP parameters of shape (len(basins), 3, 4).
    basins
        list of basin names.
    enso_simulations
        ENSO phase per [simulation][year], as integers.

    Returns
    -------
    npt.NDArray[np.uint8]
        Cyclone counts of shape (n_sim, n_years, n_basins)
    """

    mean_yearly_arrivals = expected_poisson_arrivals(poisson_params, basins)

    counts = np.zeros((n_sim, n_years, len(basins)), dtype=np.uint8)

    for sim_idx in trange(n_sim, desc="Simulating number of cyclones", disable=not show_progress):
        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]
            for b_idx, _ in enumerate(basins):
                counts[sim_idx, year_idx, b_idx] = np.random.poisson(mean_yearly_arrivals[b_idx, enso_idx])

    return counts

def enso_probs(enso_idx: int, p: float) -> npt.NDArray[np.float64]:
    """
    Return a probability vector for drawing events from ENSO phases,
    biased toward the given ENSO phase.

    Parameters
    ----------
    enso_idx
        Index of the simulated ENSO phase (0: Nina, 1: Neutral, 2: Nino).
    p
        Probability of selecting an event from the current ENSO phase.

    Returns
    -------
    npt.NDArray[np.float64]
        Array of probabilities [p_nina, p_neutral, p_nino], summing to 1.
    """
    if enso_idx not in (0, 1, 2):
        raise ValueError("Invalid ENSO index. Must be 0 (Nina), 1 (Neutral), or 2 (Nino).")

    q = (1 - p) / 2
    probs = np.full(3, q)
    probs[enso_idx] = p
    return probs


def sample_synthetic_cyclone_ids(
    n_sim: int,
    basins: list[str],
    haz_basin: npt.NDArray[np.str_],
    haz_enso: npt.NDArray[np.str_],
    haz_event_id: npt.NDArray[np.int_],
    enso_simulations: npt.NDArray[np.uint8],        # shape (n_sim, n_years)
    event_numbers: npt.NDArray[np.uint8],           # shape (n_sim, n_years, n_basins)
    loc: bool,
    p_loc: float,
    replace=True,
    show_progress=False,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.uint32]]:
    """
    Sample synthetic cyclone event IDs per simulation/year/basin with or without replacement.

    Parameters
    ----------
    basins
        Names of the cyclone basins (ordered).
    haz_basin
        Basin name for each historical event.
    haz_enso
        ENSO phase for each historical event (e.g. "Nina", "Neutral", "Nino").
    haz_event_id
        Event ID for each historical cyclone.
    
    Returns
    -------
    event_ids_flat : npt.NDArray[np.int32]
        Flat array of sampled event IDs.
    index_map : npt.NDArray[np.uint32]
        Array of shape (n_sim, n_years, n_basins, 2) containing [start_idx, count].
    """
    n_years = enso_simulations.shape[1]
    total_events = event_numbers.sum()

    event_ids_flat = np.empty(total_events, dtype=np.int32)
    index_map = np.zeros((n_sim, n_years, len(basins), 2), dtype=np.uint32)

    # Pre-index the historical event data
    event_ids_by_basin = {b: haz_event_id[haz_basin == b] for b in basins}
    enso_by_basin = {b: haz_enso[haz_basin == b] for b in basins}

    current_index = 0

    for sim_idx in trange(n_sim, desc="Sampling synthetic cyclone ids", disable=not show_progress):
        used_ids : Optional[set[int]] = set() if not replace else None

        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]
            selection_probs = enso_probs(enso_idx, p_loc) if loc else None

            for b_idx, basin in enumerate(basins):
                n_events = event_numbers[sim_idx, year_idx, b_idx]
                index_map[sim_idx, year_idx, b_idx, 0] = current_index
                index_map[sim_idx, year_idx, b_idx, 1] = n_events

                if n_events == 0:
                    continue

                all_ids_in_basin = event_ids_by_basin[basin]

                if loc:
                    enso_labels = enso_by_basin[basin]
                    sampled_phases = np.random.choice(ENSO_PHASES, size=n_events, p=selection_probs)
                    counts = Counter(sampled_phases)

                    selected_ids : list[int] = []
                    for phase, count in counts.items():
                        available = all_ids_in_basin[enso_labels == phase]
                        if not replace:
                            available = np.setdiff1d(available, list(used_ids), assume_unique=True)
                        
                        if len(available) < count:
                            raise ValueError(f"Not enough events in basin '{basin}' for ENSO phase '{phase}'.")
                        selected = np.random.choice(available, size=count, replace=replace)
                        selected_ids.extend(selected)
                        if not replace:
                            used_ids.update(selected)
                else:
                    if replace:
                        available = all_ids_in_basin
                    else:
                        available = np.setdiff1d(all_ids_in_basin, list(used_ids), assume_unique=True)
                        
                    if len(available) < n_events:
                        raise ValueError(f"Not enough events in basin '{basin}' to sample without replacement.")
                    selected_ids = np.random.choice(available, size=n_events, replace=replace)
                    if not replace:
                        used_ids.update(selected_ids)

                event_ids_flat[current_index:current_index + n_events] = selected_ids
                current_index += n_events

    return event_ids_flat, index_map


def get_poisson_parameters(haz: TropCyclone, basins: list[str], homogeneous: bool) -> npt.NDArray[np.float64]:
    """
    Load MMNHPP parameters and return as array of shape (len(basins), 3, 4).
    """
    params = np.zeros((len(basins), 3, 4)) # (basin, enso, param)
   
    if homogeneous:
        haz_basin = np.array(haz.basin)
        for b, basin in enumerate(basins):
            lambda_for_basin = haz.frequency[haz_basin == basin].sum()
            params[b, :, 0] = lambda_for_basin
        return params

    par_df = pd.read_csv("R tables/MMNHPP_par.csv")

    for b, basin in enumerate(basins):
        for e, enso in enumerate(ENSO_PHASES):
            params[b, e, :] = par_df.loc[par_df['enso'] == enso, basin].values

    # However, not all cyclones hit Australia, so we scale the parameters
    # by the proportion of cyclones that hit Australia.
    aus_cyclones = haz.select(orig=True)
    aus_cyclone_basins = np.array(aus_cyclones.basin)

    n_tracks_CLIMADA = pd.read_csv('R tables/n_tracks_CLIMADA.csv')

    for b, basin in enumerate(basins):
        n_hist_tracks = np.sum(aus_cyclone_basins == basin)
        total_tracks = n_tracks_CLIMADA[basin].values[0]
        scale = n_hist_tracks / total_tracks
    
        params[b, :, 0] *= scale
        params[b, :, 1] *= scale

    return params


def nonhomogeneous_poisson_intensity(t: float, coefs: npt.NDArray[np.float64], south_hemis=False) -> float:
    """
    Evaluate intensity function lambda(t) from seasonal model.

    Applies hemisphere-based time shift for Southern Hemisphere basins.

    Parameters
    ----------
    t
        Time in [0, 1) to evaluate lambda(t) at.
    coefs
        Coefficients of the MMNHPP intensity function.
    south_hemis
        Whether to apply the August shift for Southern Hemisphere basins.

    Returns
    -------
    float
        Value of lambda(t)
    """
    if south_hemis:
        t += 7 / 12  # August shift
    lambda_ = coefs[0] + coefs[1] * np.exp(coefs[2] * np.sin(2 * np.pi * t + coefs[3]))
    return max(lambda_, 0.0)


def expected_poisson_arrivals(poisson_params: npt.NDArray[np.float64], basins: list[str]) -> npt.NDArray[np.float64]:
    """
    Precompute expected arrival rates per basin and ENSO phase.

    Parameters
    ----------
    poisson_params
        MMNHPP parameters of shape (len(basins), 3, 4).
    basins
        list of basin names.

    Returns
    -------
    npt.NDArray[np.float64]
        Expected arrival rates of shape (len(basins), 3).
    """
    mean_yearly_arrivals = np.zeros((len(basins), len(ENSO_PHASES)))
    for b_idx, basin in enumerate(basins):
        south_hemis = basin in ['SI', 'SP']
        for enso_idx, _ in enumerate(ENSO_PHASES):
            coefs = poisson_params[b_idx, enso_idx]
            lam = integrate.quad(lambda t: nonhomogeneous_poisson_intensity(t, coefs, south_hemis), 0, 1)[0]
            mean_yearly_arrivals[b_idx, enso_idx] = lam
    return mean_yearly_arrivals


def maximum_poisson_intensities(params: npt.NDArray[np.float64], basins: list[str]) -> npt.NDArray[np.float64]:
    """
    Compute maximum of lambda(t) over [0, 1] for each (basin, enso).

    Parameters
    ----------
    params
        MMNHPP parameters of shape (len(basins), 3, 4)

    Returns
    -------
    npt.NDArray[np.float64]
        Maximum lambda values of shape (len(basins), 3)
    """
    n_basins = len(basins)
    max_intensities = np.zeros((n_basins, 3))

    for b_idx, basin in enumerate(basins):
        south_hemis = basin in ['SI', 'SP']
        for e_idx in range(3):
            coefs = params[b_idx, e_idx]
            result = optimize.minimize_scalar(
                lambda t: -nonhomogeneous_poisson_intensity(t, coefs, south_hemis),
                bounds=(0, 1),
                method='bounded'
            )
            max_intensities[b_idx, e_idx] = -result.fun

    return max_intensities


def sample_poisson_arrivals(
    n_events: int,
    coefs: npt.NDArray[np.float64],
    south_hemis: bool,
    max_intensity: float
) -> npt.NDArray[np.float32]:
    """
    Sample an arrival time from an MMNHPP via rejection sampling.

    Parameters
    ----------
    n_events
        Number of events to sample.
    coefs
        Coefficients of the MMNHPP intensity function.
    south_hemis
        Whether to apply the August shift for Southern Hemisphere basins.
    max_intensity
        Maximum intensity of the MMNHPP function.
    Returns
    -------
    npt.NDArray[np.float32]
        Array of sampled arrival times in [0, 1).
    """
    samples = np.empty(n_events, dtype=np.float32)

    for i in range(n_events):
        while True:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, max_intensity)

            if y < nonhomogeneous_poisson_intensity(x, coefs, south_hemis):
                samples[i] = x
                break
    return samples


def simulate_arrival_times_of_cyclones(
    n_sim: int,
    n_years: int,
    poisson_params: npt.NDArray[np.float64],    # shape (len(basins), 3, 4)
    basins: list[str],
    enso_simulations: npt.NDArray[np.uint8],  # shape (n_sim, n_years), int8
    event_numbers: npt.NDArray,     # shape (n_sim, n_years, n_basins), int
    show_progress=False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """
    Simulate within-season arrival times for tropical cyclones and store in a flat array.

    Parameters
    ----------
    n_sim
        Number of simulations.
    n_years
        Number of years per simulation.
    poisson_params
        MMNHPP parameters of shape (len(basins), 3, 4).
    basins
        list of basin names.
    enso_simulations
        ENSO phase per [simulation][year].
    event_numbers
        Cyclone counts per [simulation][year][basin].
    
    Returns
    -------
    arrival_times : npt.NDArray[float32]
        Flat array of all arrival times.
    index_map : npt.NDArray[uint32]
        Array of shape (n_sim, n_years, n_basins, 2) containing:
        - index_map[..., 0] = start index into arrival_times
        - index_map[..., 1] = number of events for that (sim, year, basin)
    """
    n_basins = len(basins)
    total_events = event_numbers.sum()
    
    arrival_times = np.empty(total_events, dtype=np.float32)
    index_map = np.zeros((n_sim, n_years, n_basins, 2), dtype=np.uint32)

    max_intensities = maximum_poisson_intensities(poisson_params, basins)

    current_index = 0

    for sim_idx in trange(n_sim, desc="Simulating arrival times", disable=not show_progress):
        for year_idx in range(n_years):
            enso_idx = enso_simulations[sim_idx, year_idx]

            for basin_idx, basin in enumerate(basins):
                n_events = event_numbers[sim_idx, year_idx, basin_idx]
                index_map[sim_idx, year_idx, basin_idx, 0] = current_index
                index_map[sim_idx, year_idx, basin_idx, 1] = n_events

                if n_events > 0:
                    coefs = poisson_params[basin_idx, enso_idx]
                    south_hemis = basin in ['SI', 'SP']
                    times = year_idx + sample_poisson_arrivals(n_events, coefs, south_hemis, max_intensities[basin_idx, enso_idx])
                    times.sort()
                    arrival_times[current_index:current_index + n_events] = times.astype(np.float32)
                    current_index += n_events

    return arrival_times, index_map


def create_year_loss_table_for_batch(
    sim_indices: list[int],
    enso_simulations: npt.NDArray[np.uint8],              # shape (n_sim, n_years)
    number_of_cyclones: npt.NDArray[np.uint8],           # shape (n_sim, n_years, n_basins)
    cyclone_ids_flat: npt.NDArray[np.int_],              # shape (total_events,)
    cyclone_ids_index_map: npt.NDArray[np.uint32],       # shape (n_sim, n_years, n_basins, 2)
    basins: list[str],
    arrival_times_flat: npt.NDArray[np.float32],
    arrival_times_index_map: npt.NDArray[np.uint32],     # shape (n_sim, n_years, n_basins, 2)
) -> pd.DataFrame:
    """
    Create a tidy DataFrame for a batch of simulation indices.

    Parameters
    ----------
    sim_indices
        Indices of simulations to include in this batch.
    enso_simulations
        ENSO phases for all simulations.
    number_of_cyclones
        Cyclone counts for all simulations.
    cyclone_ids_flat
        Flattened event ID array.
    cyclone_ids_index_map
        Index map for locating cyclone IDs.
    basins
        Ordered list of basin names.
    arrival_times_flat
        Flattened array of arrival times.
    arrival_times_index_map
        Index map for locating arrival times.

    Returns
    -------
    pd.DataFrame
        Tidy year loss table (losses yet to be added) for the selected simulations.
    """
    records = []
    n_years = enso_simulations.shape[1]

    for sim_idx in sim_indices:
        for year_idx in range(n_years):
            enso_phase = ENSO_PHASES[enso_simulations[sim_idx, year_idx]]

            for basin_idx, basin in enumerate(basins):
                count = number_of_cyclones[sim_idx, year_idx, basin_idx]
                if count == 0:
                    continue

                start, n = cyclone_ids_index_map[sim_idx, year_idx, basin_idx]
                ids = cyclone_ids_flat[start:start + n]

                time_start, time_n = arrival_times_index_map[sim_idx, year_idx, basin_idx]
                assert time_n == n, "Mismatch in arrival time and event ID counts"
                times = arrival_times_flat[time_start:time_start + time_n]

                for i in range(n):
                    record = {
                        "simulation": sim_idx + 1,
                        "year": year_idx + 1,
                        "event_time": times[i],
                        "basin": basin,
                        "event_id": ids[i],
                        "enso_phase": enso_phase,
                    }
                    records.append(record)

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["simulation", "year", "event_time"]).reset_index(drop=True)


@njit
def apply_decay_jit_csc_inplace(
    indptr: npt.NDArray,
    indices: npt.NDArray,
    data: npt.NDArray,
    simulation: npt.NDArray,
    event_time: npt.NDArray,
):
    """
    JIT-compiled core: applies in-place time-decay compounding to CSC matrix data,
    using simulation boundaries and event time.

    Parameters
    ----------
    indptr
        CSC indptr array (column start/end indices).
    indices
        CSC row indices for non-zero entries.
    data
        CSC data array (will be modified in place).
    simulation
        Simulation index per row (must align with matrix rows).
    event_time
        Event time per row (must align with matrix rows).
    """
    n_cols = len(indptr) - 1

    for col in range(n_cols):
        start = indptr[col]
        end = indptr[col + 1]

        last_sim = -1
        last_time = -1.0

        for i in range(start, end):
            row_idx = indices[i]
            sim = simulation[row_idx]
            time = event_time[row_idx]

            if sim != last_sim:
                last_sim = sim
                last_time = time
                continue

            tau = time - last_time
            if 0 <= tau < 1:
                theta = np.exp(-5 * tau)
                data[i] *= 1 + theta

            last_time = time

def increased_losses_from_consecutive_cyclones(
    batch_year_loss_table: pd.DataFrame,
    losses: sp.csr_matrix,
) -> sp.csr_matrix:
    """
    Efficiently apply time-decay compounding to losses using CSC format.
    Adjusts values in-place for each column (grid cell), based on simulation
    and event time of prior positive-loss events.

    Parameters
    ----------
    batch_year_loss_table
        Must contain 'simulation' and 'event_time', sorted accordingly.
    losses
        Sparse matrix of shape (n_events, n_locations).

    Returns
    -------
    sp.csr_matrix
        Adjusted sparse loss matrix.
    """
    # Convert to CSC for column-wise iteration
    losses_csc = losses.tocsc().copy()

    simulation = batch_year_loss_table["simulation"].to_numpy().astype(np.int32)
    event_time = batch_year_loss_table["event_time"].to_numpy().astype(np.float32)

    apply_decay_jit_csc_inplace(
        losses_csc.indptr,
        losses_csc.indices,
        losses_csc.data,
        simulation,
        event_time,
    )

    return losses_csc.tocsr()

def summarise_large_losses(matrix: sp.csr_matrix, sigfigs: int = 4, min_loss = 10_000) -> list[str]:
    """
    Summarise larger entries in a CSR matrix as strings of the form:
    "loc_5:12.34,loc_78:56.78"
    
    Parameters:
    - matrix: scipy.sparse.csr_matrix (rows = events, cols = loc_* values)
    - col_offset: integer to add to column indices (e.g., if loc_0 is col 0, offset=0)
    - precision: number of decimal places to keep

    Returns:
    - list of summary strings, one per row
    """
    row_summaries = []
    fmt_str = f"{{:.{sigfigs}g}}"

    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i+1]
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        summary = "|".join(
            f"{j}:{fmt_str.format(v)}"
            for j, v in zip(indices, values)
            if v >= min_loss
        )
        row_summaries.append(summary)

    return row_summaries

def assign_state_to_exposure(exp_gdf: gpd.GeoDataFrame, state_shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Add a 'state' column (based on STE_NAME21) to a GeoDataFrame of exposure points using spatial join.

    Parameters
    ----------
    exp_gdf
        GeoDataFrame with point geometries and any CRS.
    state_shapefile_path
        Path to ABS 2021 state boundary shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of the input GeoDataFrame with an added 'STE_NAME21' column indicating state.
    """
    # Load and prepare state polygons
    state_gdf = gpd.read_file(state_shapefile_path).iloc[:-1]  # Drop "Other Territories"
    state_gdf = state_gdf.to_crs("EPSG:7844")
    exp_gdf = exp_gdf.to_crs("EPSG:7844")

    exp_with_state = gpd.sjoin(exp_gdf.to_crs("EPSG:7844"), state_gdf[['STE_NAME21', 'geometry']], how='left', predicate='within')

    # Handle unmatched points using nearest fallback
    missing = exp_with_state[exp_with_state["STE_NAME21"].isna()].copy()
    missing = missing.drop(columns=['index_right', 'STE_NAME21'], errors='ignore')

    if not missing.empty:
        nearest = gpd.sjoin_nearest(
            missing,
            state_gdf[['STE_NAME21', 'geometry']],
            how='left',
            distance_col='distance'
        )
        exp_with_state.update(nearest[['STE_NAME21']])

    return exp_with_state

def save_number_of_cyclones(
    number_of_cyclones: npt.NDArray[np.uint8],
    basins: list[str],
    n_sim: int,
    n_years: int,
    output_dir: str,
    base_filename: str,
) -> None:
    """
    Save the number of simulated cyclones per simulation, year, and basin to a CSV.
    This is mostly for debugging purposes.
    The number of cyclones here will be larger than the number of cyclones in the year loss tables.
    That is because some cyclones cause no losses, so they are not included in the year loss tables.

    Parameters
    ----------
    number_of_cyclones
        Cyclone counts with shape (n_sim, n_years, n_basins).
    basins
        list of basin names (in order corresponding to last axis of number_of_cyclones).
    n_sim
        Number of simulation runs.
    n_years
        Number of years in each simulation.
    output_dir
        Directory to save the CSV file to.
    base_filename
        Base name for the output file (will append "_cyclone_counts.csv").
    """
    n_basins = len(basins)
    simulations = np.repeat(np.arange(1, n_sim + 1), n_years * n_basins)
    years = np.tile(np.repeat(np.arange(1, n_years + 1), n_basins), n_sim)
    basin_column = np.tile(basins, n_sim * n_years)
    cyclone_counts = number_of_cyclones.reshape(-1)

    cyclone_df = pd.DataFrame({
        'simulation': simulations,
        'year': years,
        'basin': basin_column,
        'number_of_cyclones': cyclone_counts
    })

    cyclone_df_wide = cyclone_df.pivot(
        index=['simulation', 'year'],
        columns='basin',
        values='number_of_cyclones'
    ).reset_index()

    cyclone_df_wide.columns = ['simulation', 'year'] + [f"Number of cyclones ({basin} basin)" for basin in basins]
    cyclone_df_wide.to_csv(os.path.join(output_dir, f"{base_filename}_cyclone_counts.csv"), index=False)

def generate_year_loss_tables(
    haz: TropCyclone,
    loss_catalogues: dict[str, sp.csr_matrix],
    loss_catalogue_index: dict[int, int],
    homogeneous: bool,
    n_sim: int,
    n_years: int,
    starting_enso="Nino",
    loc=False,
    p_loc=0.5,
    intensity=False,
    damage=False,
    regions: Optional[pd.Series] = None,
    batch_size=10_000,
    output_dir="Outputs",
    save_large_losses=True,
    replace=True,
    show_progress=False,
    seed=None,
    save_poisson_debug=False,
) -> str:
    if seed:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{homogeneous}_{n_sim}_{n_years}_{loc}_{p_loc}_{intensity}_{damage}"

    print(f"{base_filename}: 1/5) Simulating ENSO time series", flush=True)
    enso_simulations: npt.NDArray[np.uint8] = simulate_enso_time_series(
        n_sim, n_years, starting_enso, show_progress
    )

    print(f"{base_filename}: 2/5) Simulating number of cyclones", flush=True)
    basins: list[str] = list(np.sort(np.unique(haz.basin)))
    poisson_params = get_poisson_parameters(haz, basins, homogeneous=homogeneous)
    number_of_cyclones: npt.NDArray[np.uint8] = simulate_number_of_cyclones(
        n_sim, n_years, poisson_params, basins, enso_simulations, show_progress
    )

    if save_poisson_debug:
        save_number_of_cyclones(
            number_of_cyclones,
            basins,
            n_sim,
            n_years,
            output_dir,
            base_filename
        )
        
    print(f"{base_filename}: 3/5) Simulating arrival times of cyclones", flush=True)
    arrival_times_flat, arrival_times_index_map = simulate_arrival_times_of_cyclones(
        n_sim,
        n_years,
        poisson_params,
        basins,
        enso_simulations,
        number_of_cyclones,
        show_progress
    )

    print(f"{base_filename}: 4/5) Sampling synthetic cyclone IDs", flush=True)

    haz_basin = np.array(haz.basin)
    haz_event_id = np.array(haz.event_id)
    haz_enso: npt.NDArray[np.str_] = enso_phases_of_historical_tracks(haz)

    cyclone_ids_flat, cyclone_ids_index_map = sample_synthetic_cyclone_ids(
        n_sim,
        basins,
        haz_basin,
        haz_enso,
        haz_event_id,
        enso_simulations,
        number_of_cyclones,
        loc,
        p_loc,
        replace=replace,
        show_progress=show_progress
    )

    print(f"{base_filename}: 5/5) Creating year loss table in batches", flush=True)
    start_time = time()

    sim_batches = [list(range(i, min(i + batch_size, n_sim))) for i in range(0, n_sim, batch_size)]
    
    if regions is not None:
        # Map regions to column indices
        region_names = sorted(regions.unique())
        region_col_map = {region: np.where(regions.values == region)[0] for region in region_names}

    for batch_num, sim_indices in enumerate(tqdm(sim_batches, desc=f"{base_filename}: Calculate losses of resampled hazards", disable=not show_progress)):

        batch_year_loss_table = create_year_loss_table_for_batch(
            sim_indices=sim_indices,
            enso_simulations=enso_simulations,
            number_of_cyclones=number_of_cyclones,
            cyclone_ids_flat=cyclone_ids_flat,
            cyclone_ids_index_map=cyclone_ids_index_map,
            basins=basins,
            arrival_times_flat=arrival_times_flat,
            arrival_times_index_map=arrival_times_index_map
        )

        if batch_year_loss_table.empty:
            continue

        # Compute losses
        if intensity:
            enso_phases = batch_year_loss_table["enso_phase"].values
            row_list = []
            for eid, phase in zip(batch_year_loss_table.event_id, enso_phases):
                row_idx = loss_catalogue_index[eid]
                row = loss_catalogues[phase].getrow(row_idx)
                row_list.append(row)
            losses = sp.vstack(row_list, format="csr")
        else:
            rows = [loss_catalogue_index[eid] for eid in batch_year_loss_table.event_id]
            losses = loss_catalogues["No adjustment"][rows]
        
        # Only keep the rows which have a positive loss
        positive = np.array((losses.sum(axis=1) > 0)).ravel()
        batch_year_loss_table = batch_year_loss_table.loc[positive, :]
        losses = losses[positive, :]

        if damage:
            losses = increased_losses_from_consecutive_cyclones(batch_year_loss_table, losses)

        batch_year_loss_table["total_loss"] = losses.sum(axis=1)

        if regions is not None:
            # Sum loss by region and insert as new columns
            for region in region_names:
                cols = region_col_map[region]
                region_loss = losses[:, cols].sum(axis=1).A1  # Convert to 1D array
                batch_year_loss_table[f"{region.lower().replace(' ', '_')}_loss"] = region_loss

        if save_large_losses:
            batch_year_loss_table["larger_losses"] = summarise_large_losses(losses)

        # Append batch to CSV
        year_loss_table_file = os.path.join(output_dir, f"{base_filename}.csv")
        write_header = batch_num == 0  # only write header on first batch
        batch_year_loss_table.to_csv(year_loss_table_file, index=False, mode="w" if write_header else "a", header=write_header)

    print(f"{base_filename}: All batches processed in {time() - start_time:.2f} seconds", flush=True)

    # Convert the CSV to parquet
    df = pd.read_csv(year_loss_table_file)
    df['year'] = df['year'].astype(np.uint32)
    df['simulation'] = df['simulation'].astype(np.uint32)
    df['event_time'] = df['event_time'].astype(np.float32)
    df['event_id'] = df['event_id'].astype(np.int32)
    df['total_loss'] = df['total_loss'].astype(np.float32)
    df['enso_phase'] = df['enso_phase'].astype("category")
    df['basin'] = df['basin'].astype("category")
    df.to_parquet(year_loss_table_file.replace('.csv', '.parquet'), compression="zstd", index=False)
    os.remove(year_loss_table_file)
    print(f"{base_filename}: Year loss table saved as parquet", flush=True)

    return base_filename
