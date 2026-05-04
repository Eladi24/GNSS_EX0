import numpy as np
import gnss_lib_py as glp
from datetime import datetime, timezone
from gnss_lib_py.utils.ephemeris_downloader import load_ephemeris
from gnss_lib_py.parsers.rinex_nav import get_time_cropped_rinex
from gnss_lib_py.utils.sv_models import find_sv_states
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.parsers.rinex_nav import RinexNav
import corrections as cor
from solver import ecef_to_geodetic, ecef_to_azel

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
    "G": ["C1C"],        # L5 (C5Q) has large satellite-specific DCBs on Samsung; L1 only
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

def dt_to_gps_millis(dt: datetime) -> float:
    """Convert GPS time datetime directly to GPS millis without leap second shifts."""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    gps_epoch = datetime(1980, 1, 6)
    return (dt - gps_epoch).total_seconds() * 1000.0

def crop_ephemeris(epochs: list, gps_millis: float):

    all_sat_ids = set()
    for epoch in epochs:
        for sat in epoch['satellites']:
            all_sat_ids.add(sat['id'])
    try:
        ephem = get_time_cropped_rinex(
            gps_millis          = gps_millis,
            satellites          = list(all_sat_ids),
            ephemeris_directory = EPHEMERIS_DIR,
        )
        print(f"  Ephemeris cropped once for {len(all_sat_ids)} satellites")
        return ephem
    except Exception as e:
        print(f"  Warning: could not crop ephemeris: {e}")
        return None

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


def get_ephemeris(timestamp) -> tuple[NavData, list]:
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

    gps_millis = dt_to_gps_millis(timestamp)
    file_paths = load_ephemeris(
        file_type      = "rinex_nav",
        gps_millis     = gps_millis,
        constellations = ["gps", "glonass", "galileo", "beidou"],
        verbose        = True,
    )
    print(f"Downloaded nav files: {file_paths}")
    return RinexNav(file_paths), file_paths

def _eval_clock_poly(nav_data, col: int, gps_millis_tx: float) -> float | None:
    """
    Evaluate the satellite clock polynomial at gps_millis_tx.

    gnss_lib_py NavData uses:
      SVclockBias (s), SVclockDrift (s/s), SVclockDriftRate (s/s²)
      gps_millis  — absolute GPS epoch of t_oc in milliseconds (same time
                    base as gps_millis_tx, so dt is just their difference).
    Returns b_sv_m in metres (add to pseudorange), or None.
    """
    try:
        bias   = float(nav_data['SVclockBias',      col])
        drift  = float(nav_data['SVclockDrift',     col])
        drift2 = float(nav_data['SVclockDriftRate', col])
        t_ref  = float(nav_data['gps_millis',       col])  # ms at t_oc epoch
        if any(np.isnan([bias, drift, drift2, t_ref])):
            return None
        dt = (gps_millis_tx - t_ref) / 1000.0   # ms → seconds
        if dt >  302400: dt -= 604800
        if dt < -302400: dt += 604800
        return (bias + drift * dt + drift2 * dt**2) * SPEED_OF_LIGHT
    except Exception:
        return None


def compute_clock_correction(sv_states, ephem,
                              sv_states_idx: int,
                              gnss_id_str: str, sv_id_int: int,
                              gps_millis_tx: float) -> float | None:
    """
    Compute satellite clock correction in metres from RINEX nav polynomial
    (af0 + af1·dt + af2·dt²). gnss_lib_py's b_sv_m is bypassed because it
    includes undocumented delay terms reaching ±100 km.

    Strategy:
      1. Try sv_states at sv_states_idx directly — fastest; works when
         find_sv_states preserves original nav fields in-place.
      2. Search ephem by gnss_id + sv_id — robust to column-order differences
         between sv_states and the cropped ephemeris.

    Returns correction in metres (add to pseudorange), or None.
    """
    # Fast path: sv_states preserves original nav fields at the same index
    b = _eval_clock_poly(sv_states, sv_states_idx, gps_millis_tx)
    if b is not None:
        return b

    # Slow path: search ephem by satellite identity.
    # ephem['gnss_id', j] returns a numpy 0-d array → .item() extracts the scalar.
    for j in range(ephem.shape[1]):
        try:
            if (str(ephem['gnss_id', j].item()) != gnss_id_str or
                    int(ephem['sv_id',   j].item()) != sv_id_int):
                continue
            b = _eval_clock_poly(ephem, j, gps_millis_tx)
            if b is not None:
                return b
        except Exception:
            continue

    return None


def compute_sat_position(epoch, ephem, rx_pos=None) -> list:
    """
    For a single epoch, compute the ECEF position and clock bias
    of every satellite that has a valid pseudorange.

    Parameters
    ----------
    epoch : dict
        One epoch from parse_rinex_file() output
    ephem : RinexNav
        Parsed ephemeris data from crop_ephemeris()
    rx_pos : np.ndarray, optional
        Receiver ECEF position (if available)

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

    gps_millis_rx = dt_to_gps_millis(timestamp)

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

    if rx_pos is not None:
        rx_lat, rx_lon, rx_alt = ecef_to_geodetic(*rx_pos)
    else:
        rx_lat = rx_lon = rx_alt = None

    # Map sat_id → which pseudorange observable was selected (for ISB grouping)
    pr_obs_type_map = {}
    for sat in epoch['satellites']:
        pr_obs = None
        priorities = PR_PRIORITY.get(sat['sys'], ["C1C"])
        for code in priorities:
            if sat.get(code) is not None:
                pr_obs = code
                break
        if pr_obs is not None:
            pr_obs_type_map[sat['id']] = pr_obs

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
        vx = float(sv_states['vx_sv_mps', i])
        vy = float(sv_states['vy_sv_mps', i])
        vz = float(sv_states['vz_sv_mps', i])
        pr = valid_sats[sat_id]

        if any(np.isnan([x, y, z])):
            print(f"  Skipping {sat_id} — NaN position")
            continue

        # Correct satellite position for individual transmission time difference
        # find_sv_states was evaluated at the average transmission time.
        dt_s = (avg_pseudorange - pr) / SPEED_OF_LIGHT
        x += vx * dt_s
        y += vy * dt_s
        z += vz * dt_s

        # Relativistic clock correction: -2 * (r_sat \cdot v_sat) / c
        rel_corr = -2.0 * (x * vx + y * vy + z * vz) / SPEED_OF_LIGHT

        # Sagnac correction: rotate satellite ECEF by Earth's rotation
        # during signal travel time so receiver and satellite share one frame.
        EARTH_ROT_RATE = 7.292115e-5          # rad/s
        tau = pr / SPEED_OF_LIGHT             # ~0.067 s
        x_s = x + EARTH_ROT_RATE * y * tau
        y_s = y - EARTH_ROT_RATE * x * tau
        x, y = x_s, y_s

        gps_millis_tx_sat = gps_millis_rx - (pr / SPEED_OF_LIGHT) * 1000.0

        # Always derive clock correction from nav polynomial — gnss_lib_py's
        # b_sv_m includes undocumented hardware delay terms that reach ±100 km.
        b = compute_clock_correction(sv_states, ephem, i, gnss_id, sv_id,
                                     gps_millis_tx_sat)
        MAX_CLOCK_M = 500_000.0  # 1.67 ms × c; beyond this the poly eval is wrong
        if b is None:
            print(f"  Skipping {sat_id} — no clock correction in nav data")
            continue

        b += rel_corr

        if abs(b) > MAX_CLOCK_M:
            print(f"  Skipping {sat_id} — implausible clock correction ({b/1e3:+.0f} km)")
            continue
        b_valid = True

        # Atmospheric corrections (only if we have a rough position)
        iono_corr  = 0.0
        tropo_corr = 0.0
        el         = None
        if rx_pos is not None:
            sv_xyz = np.array([x, y, z])
            el, az = ecef_to_azel(rx_pos, sv_xyz)
            el_rad = np.radians(el)
            az_rad = np.radians(az)
            rx_lat_rad = np.radians(rx_lat)
            rx_lon_rad = np.radians(rx_lon)

            tropo_corr = cor.saastamoinen_tropo(el_rad, rx_alt)

            if cor.KLOBUCHAR_ALPHA is not None:
                iono_corr = cor.klobuchar_iono(
                    gps_millis_rx, el_rad, az_rad,
                    rx_lat_rad, rx_lon_rad,
                    cor.KLOBUCHAR_ALPHA, cor.KLOBUCHAR_BETA
                )

        doppler, doppler_lambda = doppler_obs.get(sat_id, (None, None))

        sat_positions.append({
            'id':             sat_id,
            'pseudorange':    pr,
            'corrected_pr':   pr + b + iono_corr + tropo_corr,
            'x_sv_m':         x,
            'y_sv_m':         y,
            'z_sv_m':         z,
            'b_sv_m':         b,
            'b_sv_m_valid':   b_valid,
            'elevation_deg':  el,
            'pr_obs_type':    pr_obs_type_map.get(sat_id, 'C1C'),
            'vx_sv_mps':      vx,
            'vy_sv_mps':      vy,
            'vz_sv_mps':      vz,
            'doppler':        doppler,
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
    