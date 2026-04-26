import numpy as np
import gnss_lib_py as glp
from datetime import datetime, timezone
from gnss_lib_py.utils.ephemeris_downloader import load_ephemeris
from gnss_lib_py.utils.time_conversions import datetime_to_gps_millis
from gnss_lib_py.parsers.rinex_nav import get_time_cropped_rinex
from gnss_lib_py.utils.sv_models import find_sv_states
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.parsers.rinex_nav import RinexNav

SPEED_OF_LIGHT = 299_792_458.0
EPHEMERIS_DIR  = "data/ephemeris"

GNSS_ID_MAP = {
    "gps":     "G",
    "glonass": "R",
    "galileo": "E",
    "beidou":  "C",
}

# Ordered preference of pseudorange observable per constellation
PR_PRIORITY = {
    "G": ["C1C", "C5Q"],
    "R": ["C1C"],
    "E": ["C1C", "C5Q"],
    "C": ["C2I", "C5Q"],
}

GPS_L1_FREQ = 1_575_420_000.0                    # Hz
GPS_L5_FREQ = 1_176_450_000.0                    # Hz
GPS_L1_LAMBDA = SPEED_OF_LIGHT / GPS_L1_FREQ    # ~0.1903 m
GPS_L5_LAMBDA = SPEED_OF_LIGHT / GPS_L5_FREQ    # ~0.2548 m

DOPPLER_PRIORITY = {
    "G": [("D1C", GPS_L1_LAMBDA), ("D5Q", GPS_L5_LAMBDA)],
    "R": [("D1C", GPS_L1_LAMBDA)],   # approximate, GLONASS is FDMA
    "E": [("D1C", GPS_L1_LAMBDA), ("D5Q", GPS_L5_LAMBDA)],
    "C": [("D5Q", GPS_L5_LAMBDA), ("D2I", SPEED_OF_LIGHT / 1_207_140_000.0)],
}

def select_doppler(sat: dict) -> tuple[float, float] | tuple[None, None]:
    """
    Return (doppler_hz, wavelength_m) for the best available Doppler observable.
    Returns (None, None) if no Doppler available.
    """
    priorities = DOPPLER_PRIORITY.get(sat['sys'], [("D1C", GPS_L1_LAMBDA)])
    for field, wavelength in priorities:
        val = sat.get(field)
        if val is not None:
            return val, wavelength
    return None, None

def select_pseudorange(sat: dict) -> float | None:
    """Return the best available pseudorange for this satellite."""
    priorities = PR_PRIORITY.get(sat['sys'], ["C1C", "C5Q", "C2I"])
    for code in priorities:
        val = sat.get(code)
        if val is not None:
            return val
    return None


def get_ephemeris(timestamp) -> RinexNav:
    """
    Downloads broadcast RINEX navigation file for the date of the recording
    and returns a parsed NavData object.

    Parameters
    ----------
    timestamp : datetime
        Any epoch timestamp from your RINEX file

    Returns
    -------
    nav : RinexNav
        Parsed navigation data for all constellations on that date
    """
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    gps_millis = datetime_to_gps_millis(timestamp)
    file_paths = load_ephemeris(
        file_type      = "rinex_nav",
        gps_millis     = gps_millis,
        constellations = ["gps", "glonass", "galileo", "beidou"],
        verbose        = True,
    )
    print(f"Downloaded nav files: {file_paths}")
    return RinexNav(file_paths)


def compute_sat_position(epoch, nav) -> list:
    """
    For a single epoch, compute the ECEF position and clock bias
    of every satellite that has a valid pseudorange.

    Parameters
    ----------
    epoch : dict
        One epoch from parse_rinex_file() output
    nav : RinexNav
        Parsed navigation data from get_ephemeris()

    Returns
    -------
    sat_positions : list of dicts
        One entry per satellite with keys:
        id, pseudorange, corrected_pr, x_sv_m, y_sv_m, z_sv_m, b_sv_m

        Note: b_sv_m follows gnss_lib_py convention (add to pseudorange).
              corrected_pr = pseudorange + b_sv_m is pre-computed for solver use.
    """
    timestamp = epoch['timestamp']
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    gps_millis_rx = datetime_to_gps_millis(timestamp)

    # Select best pseudorange per satellite using constellation-specific priority
    valid_sats = {}
    for sat in epoch['satellites']:
        pr = select_pseudorange(sat)
        if pr is not None:
            valid_sats[sat['id']] = pr
    if not valid_sats:
        return []
    
    doppler_obs = {}
    for sat in epoch['satellites']:
        d, lam = select_doppler(sat)
        if d is not None:
            doppler_obs[sat['id']] = (d, lam)
    # Crop ephemeris to the closest entry before measurement time
    try:
        ephem = get_time_cropped_rinex(
            gps_millis          = gps_millis_rx,
            satellites          = list(valid_sats.keys()),
            ephemeris_directory = EPHEMERIS_DIR,
        )
    except Exception as e:
        print(f"  Warning: could not crop ephemeris: {e}")
        return []

    # Shift receive time back by average signal travel time (~67 ms)
    avg_pseudorange = np.mean(list(valid_sats.values()))
    travel_time_ms  = (avg_pseudorange / SPEED_OF_LIGHT) * 1000.0
    gps_millis_tx   = gps_millis_rx - travel_time_ms

    # Compute satellite ECEF positions and clock biases at transmission time
    try:
        sv_states = find_sv_states(gps_millis_tx, ephem)
    except Exception as e:
        print(f"  Warning: find_sv_states failed: {e}")
        return []

    sat_positions = []
    for i in range(sv_states.shape[1]):
        gnss_id = str(sv_states['gnss_id', i].item())
        sv_id   = int(sv_states['sv_id', i].item())
        sat_id  = f"{GNSS_ID_MAP.get(gnss_id, '?')}{sv_id:02d}"

        if sat_id not in valid_sats:
            continue

        x  = float(sv_states['x_sv_m', i])
        y  = float(sv_states['y_sv_m', i])
        z  = float(sv_states['z_sv_m', i])
        b  = float(sv_states['b_sv_m', i])
        pr = valid_sats[sat_id]

        if any(np.isnan([x, y, z])):
            print(f"  Skipping {sat_id} — NaN position")
            continue

        # gnss_lib_py convention: b_sv_m is a correction to ADD.
        # Fall back to 0 for constellations where it returns NaN (e.g. Galileo).
        b_valid = not np.isnan(b)
        if not b_valid:
            b = 0.0  # only for storage, corrected_pr won't be used if invalid
        doppler, doppler_lambda = doppler_obs.get(sat_id, (None, None))
        sat_positions.append({
            'id':           sat_id,
            'pseudorange':  pr,
            'corrected_pr': pr + b,
            'x_sv_m':       x,
            'y_sv_m':       y,
            'z_sv_m':       z,
            'b_sv_m':       b,
            'b_sv_m_valid': b_valid,
            'vx_sv_mps':    float(sv_states['vx_sv_mps', i]),
            'vy_sv_mps':    float(sv_states['vy_sv_mps', i]),
            'vz_sv_mps':    float(sv_states['vz_sv_mps', i]),
            'doppler':      doppler,
            'doppler_lambda': doppler_lambda,
        })

    return sat_positions


def filter_by_elevation(sat_positions: list, rx_ecef: np.ndarray,
                         min_elevation_deg: float = 15.0) -> list:
    """
    Remove satellites below the elevation mask angle.
    Requires a rough receiver position (e.g. from GPS-only pass).

    Parameters
    ----------
    sat_positions : list of dicts
    rx_ecef : np.ndarray
        Approximate receiver ECEF position in metres
    min_elevation_deg : float
        Elevation cutoff in degrees (default 15°)

    Returns
    -------
    filtered : list of dicts
    """
    from solver import ecef_to_azel  # avoid circular import at module level

    filtered = []
    for s in sat_positions:
        sv_xyz = np.array([s['x_sv_m'], s['y_sv_m'], s['z_sv_m']])
        el, _  = ecef_to_azel(rx_ecef, sv_xyz)
        if el >= min_elevation_deg:
            filtered.append(s)
        else:
            print(f"  Dropping {s['id']} — elevation {el:.1f}° below {min_elevation_deg}° mask")
    return filtered