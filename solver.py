"""
Positioning and Navigation Solver Module.
Provides least-squares algorithms to compute position, velocity, and time from GNSS observables.
"""

import numpy as np

SPEED_OF_LIGHT      = 299_792_458.0
CONSTELLATION_ORDER = ['G', 'R', 'E', 'C']

def _guess_initial_position(sat_xyz: np.ndarray) -> np.ndarray:
    """
    General initial receiver position estimate — works anywhere on Earth.
    Project the centroid of the visible satellite constellation down to
    Earth's surface. No geographic assumptions.

    Args:
        sat_xyz (np.ndarray): Nx3 array of satellite ECEF coordinates.

    Returns:
        np.ndarray: Initial guess for receiver ECEF position [x, y, z].
    """
    centroid = np.mean(sat_xyz, axis=0)
    r = np.linalg.norm(centroid)
    if r < 1.0:
        return np.array([6_378_137.0, 0.0, 0.0])
    return centroid / r * 6_371_000.0   # scale to mean Earth radius


def least_squares_position(sat_positions: list,
                            initial_pos: np.ndarray = None,
                            height_constraint_m: float = None,
                            ) -> tuple[np.ndarray, float, np.ndarray, list]:
    """
    Robust Least-Squares Positioning with Outlier Rejection.

    Args:
        sat_positions (list): List of dictionaries containing satellite ephemeris and observables.
        initial_pos (np.ndarray, optional): Prior ECEF position [x, y, z] to linearize about.
        height_constraint_m (float, optional): Optional altitude in metres for soft constraint.

    Returns:
        tuple: (pos_ecef, dt_seconds, post_fit_residuals, outlier_ids)
            - pos_ecef: np.ndarray [x, y, z] in metres, or None if failed.
            - dt_seconds: Receiver clock bias in seconds, or None if failed.
            - post_fit_residuals: np.ndarray of residuals, or None if failed.
            - outlier_ids: list of rejected satellite PRNs.
    """
    supported = [s for s in sat_positions
                 if s['id'][0] in CONSTELLATION_ORDER
                 and s.get('b_sv_m_valid', True)]

    OUTLIER_THRESHOLD_M = 3000.0
    MIN_SATS            = 4

    if len(supported) < MIN_SATS:
        print(f"  Too few satellites ({len(supported)} < {MIN_SATS}) — skipping")
        return None, None, None, []

    pos, dt, residuals, used_sats = _solve_once(supported, initial_pos,
                                                height_constraint_m=height_constraint_m)
    if pos is None:
        return None, None, None, []

    # Single outlier pass: remove satellites whose residual exceeds threshold,
    # then re-solve once.
    outlier_ids = []
    outliers = [i for i, r in enumerate(residuals) if abs(r) > OUTLIER_THRESHOLD_M]
    if outliers:
        for i in sorted(outliers, reverse=True):
            sat = used_sats[i]
            print(f"  Outlier: dropping {sat['id']} (residual {residuals[i]:+.1f} m)")
            outlier_ids.append(sat['id'])
            supported.pop(i)
        if len(supported) < MIN_SATS:
            print("  Too few satellites after outlier removal — solution failed")
            return None, None, None, outlier_ids
        pos, dt, residuals, used_sats = _solve_once(supported, initial_pos,
                                                    height_constraint_m=height_constraint_m)
        if pos is None:
            return None, None, None, outlier_ids

    sys_used = list(dict.fromkeys(s['id'][0] for s in used_sats))
    print(f"  Using {len(used_sats)} satellites ({', '.join(sys_used)})")
    print(f"  Post-fit residuals:")
    for s, r in zip(used_sats, residuals):
        el = s.get('elevation_deg')
        el_str = f"{el:5.1f}°" if el is not None else "   N/A"
        print(f"    {s['id']}  el={el_str}  resid={r:+8.1f} m")
    return pos, dt, residuals, outlier_ids


def _solve_once(supported: list,
                initial_pos: np.ndarray = None,
                height_constraint_m: float = None) -> tuple:
    """
    Core iterative least-squares solver for a single epoch.

    Args:
        supported (list): Filtered list of satellites.
        initial_pos (np.ndarray, optional): Apriori position estimate.
        height_constraint_m (float, optional): Virtual observation of altitude.

    Returns:
        tuple: (pos_ecef, dt_m, residuals, used_sats) or (None, None, None, []).
    """

    # ISB groups: one bias parameter per (constellation, pr_obs_type) pair that
    # has >= 2 satellites. The first available group acts as the reference clock
    # to prevent design matrix singularity when GPS is not present.
    group_counts: dict[tuple, int] = {}
    for s in supported:
        g = (s['id'][0], s.get('pr_obs_type', 'C1C'))
        group_counts[g] = group_counts.get(g, 0) + 1

    # Build ordered ISB list: all non-ref groups with >= 2 sats.
    # Deterministic order: constellation order first, then obs type alphabetically.
    all_groups = []
    for sys in CONSTELLATION_ORDER:
        obs_types = sorted(set(s.get('pr_obs_type', 'C1C')
                               for s in supported if s['id'][0] == sys))
        for obs in obs_types:
            g = (sys, obs)
            if group_counts.get(g, 0) > 0:
                all_groups.append(g)

    if not all_groups:
        return None, None, None, []

    ref_group = all_groups[0]
    
    # Drop isolated satellites (< 2 in group) to prevent them from corrupting the ref clock
    supported = [s for s in supported if 
                 (s['id'][0], s.get('pr_obs_type', 'C1C')) == ref_group or 
                 group_counts[(s['id'][0], s.get('pr_obs_type', 'C1C'))] >= 2]

    present_groups = [g for g in all_groups if g != ref_group and group_counts[g] >= 2]

    n_unknowns = 4 + len(present_groups)

    if len(supported) < n_unknowns or len(supported) < 4:
        return None, None, None, []

    # Map each satellite to its ISB index (-1 = reference, no ISB).
    sat_isb_idx = []
    for s in supported:
        g = (s['id'][0], s.get('pr_obs_type', 'C1C'))
        sat_isb_idx.append(present_groups.index(g) if g in present_groups else -1)

    sat_xyz      = np.array([[s['x_sv_m'], s['y_sv_m'], s['z_sv_m']]
                              for s in supported])
    pseudoranges = np.array([s['corrected_pr'] for s in supported])

    state = np.zeros(n_unknowns)
    if initial_pos is not None:
        state[:3] = initial_pos
    else:
        state[:3] = _guess_initial_position(sat_xyz)
    rough_ranges = np.linalg.norm(sat_xyz - state[:3], axis=1)
    state[3] = float(np.mean(pseudoranges - rough_ranges))

    for iteration in range(10):
        x, y, z = state[0], state[1], state[2]
        cdt_gps  = state[3]
        diff   = sat_xyz - np.array([x, y, z])
        ranges = np.linalg.norm(diff, axis=1)
        clock_bias = np.array([
            cdt_gps + (state[4 + k] if k >= 0 else 0.0)
            for k in sat_isb_idx
        ])
        residuals  = pseudoranges - (ranges + clock_bias)
        unit_vecs  = -diff / ranges[:, np.newaxis]
        # Design matrix H: [Unit vectors | Ref Clock | ISB1 | ISB2 ... ]
        H          = np.zeros((len(supported), n_unknowns))
        H[:, :3]   = unit_vecs
        H[:, 3]    = 1.0
        for i, k in enumerate(sat_isb_idx):
            if k >= 0:
                H[i, 4 + k] = 1.0
        # Elevation-weighted LS: down-weight low-elevation sats.
        # σ ∝ 1/sin(el)  →  weight = sin(el).  Default 30° when unknown.
        elevations = np.array([
            np.radians(max(s.get('elevation_deg') or 30.0, 5.0))
            for s in supported
        ])
        W = np.diag(np.sin(elevations))

        # Soft altitude constraint: adds one pseudo-observation h(x)=h_target.
        if height_constraint_m is not None:
            r = np.linalg.norm(state[:3])
            if r > 1.0:
                _, _, alt_cur = ecef_to_geodetic(state[0], state[1], state[2])
                h_resid = height_constraint_m - alt_cur
                upward  = state[:3] / r
                h_row   = np.zeros(n_unknowns)
                h_row[:3] = upward
                H         = np.vstack([H, h_row])
                residuals = np.append(residuals, h_resid)
                W         = np.diag(np.append(np.diag(W), 1.0))

        # Solve Normal Equations: (H^T W H) delta = H^T W residuals
        delta  = np.linalg.lstsq(W @ H, W @ residuals, rcond=None)[0]
        state += delta
        if np.linalg.norm(delta[:3]) < 1e-4:
            break

    if present_groups:
        print(f"  Inter-system biases (Reference: {ref_group[0]}_{ref_group[1]}):")
        for k, (sys, obs) in enumerate(present_groups):
            isb_m = state[4 + k]
            print(f"    → {sys}_{obs}: {isb_m/SPEED_OF_LIGHT*1e9:.1f} ns  ({isb_m:+.0f} m)")

    return state[:3], state[3] / SPEED_OF_LIGHT, residuals[:len(supported)], supported

def least_squares_velocity(sat_positions: list, pos_ecef: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Estimate receiver ECEF velocity and clock drift from Doppler measurements.
    Equations are linear — solved in one step (no iteration needed).

    Parameters
    ----------
    sat_positions : list of dicts
        Must include: doppler, vx_sv_mps, vy_sv_mps, vz_sv_mps, b_sv_m_valid
    pos_ecef : np.ndarray
        Receiver ECEF position from least_squares_position()

    Returns
    -------
    vel : np.ndarray shape (3,)
        Receiver ECEF velocity [vx, vy, vz] in m/s
    f_dot : float
        Receiver clock drift in m/s
    """
    # Receiver clock drift is common to all constellations (ISB time-derivative is 0).
    # We can use all available satellites without any additional unknowns.
    valid = [s for s in sat_positions
             if s.get('doppler') is not None
             and s.get('doppler_lambda') is not None
             and s.get('b_sv_m_valid', True)]
    if len(valid) < 4:
        print(f"  Not enough satellites for velocity solution "
              f"(have {len(valid)}, need 4)")
        for s in sat_positions:
            reason = []
            if s.get('doppler') is None:      reason.append("no Doppler")
            if not s.get('b_sv_m_valid',True): reason.append("no clock")
            if reason:
                print(f"    {s['id']}: {', '.join(reason)}")
        return None, None
    
    
    print(f"  Velocity: using {len(valid)} satellites")
    # Convert Doppler (Hz) to range-rate (m/s): range_rate = -D1C × λ
    range_rates = np.array([-s['doppler'] * s['doppler_lambda'] for s in valid])

    sat_xyz = np.array([[s['x_sv_m'], s['y_sv_m'], s['z_sv_m']] for s in valid])
    sat_vel = np.array([[s['vx_sv_mps'], s['vy_sv_mps'], s['vz_sv_mps']] for s in valid])

    # Unit vectors from receiver to each satellite
    diff = sat_xyz - pos_ecef
    ranges = np.linalg.norm(diff, axis=1)
    unit_vecs = diff / ranges[:, np.newaxis]

    # Range-rate from satellite motion projected onto LOS
    sat_range_rates = np.sum(sat_vel * unit_vecs, axis=1)

    # Observation: range_rate = sv_range_rate - rx_range_rate + f_dot
    # → rx_range_rate - f_dot = sv_range_rate - range_rate
    obs = sat_range_rates - range_rates
    H = np.hstack([unit_vecs, np.ones((len(valid), 1))])  # [LOS | clock drift]

    # Elevation-weighted LS: down-weight noisy low-elevation Doppler measurements
    elevations = np.array([
        np.radians(max(s.get('elevation_deg') or 30.0, 5.0))
        for s in valid
    ])
    W = np.diag(np.sin(elevations))

    # One-step linear solve (W @ H @ x = W @ obs)
    result = np.linalg.lstsq(W @ H, W @ obs, rcond=None)[0]
    vel = result[:3]
    f_dot = result[3]
    return vel, f_dot

def ecef_to_geodetic(x, y, z) -> tuple[float, float, float]:
    """
    Convert ECEF (metres) to geodetic (lat, lon, alt).
    Uses Bowring's iterative method with WGS-84 ellipsoid parameters.

    Args:
        x (float): ECEF X coordinate in metres.
        y (float): ECEF Y coordinate in metres.
        z (float): ECEF Z coordinate in metres.

    Returns:
        tuple: (latitude_deg, longitude_deg, altitude_m).
    """
    a  = 6_378_137.0
    e2 = 6.6943799901e-3

    lon = np.degrees(np.arctan2(y, x))
    p   = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))

    for _ in range(10):
        N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    N   = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), lon, alt


def ecef_to_azel(rx_xyz: np.ndarray,
                 sv_xyz: np.ndarray) -> tuple[float, float]:
    """
    Compute elevation and azimuth (degrees) from receiver to satellite.

    Args:
        rx_xyz (np.ndarray): Receiver ECEF coordinates [x, y, z] in metres.
        sv_xyz (np.ndarray): Satellite ECEF coordinates [x, y, z] in metres.

    Returns:
        tuple: (elevation_deg, azimuth_deg).
    """
    lat, lon, _ = ecef_to_geodetic(*rx_xyz)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    sin_lat, cos_lat = np.sin(lat_r), np.cos(lat_r)
    sin_lon, cos_lon = np.sin(lon_r), np.cos(lon_r)

    diff = sv_xyz - rx_xyz
    e =  -sin_lon           * diff[0] + cos_lon           * diff[1]
    n =  -sin_lat * cos_lon * diff[0] - sin_lat * sin_lon * diff[1] + cos_lat * diff[2]
    u =   cos_lat * cos_lon * diff[0] + cos_lat * sin_lon * diff[1] + sin_lat * diff[2]

    el = np.degrees(np.arctan2(u, np.sqrt(e**2 + n**2)))
    az = np.degrees(np.arctan2(e, n)) % 360
    return el, az


def validate_with_gsv(position_ecef: np.ndarray, sat_positions: list):
    """
    Compare computed satellite geometry against NMEA GSV ground truth.

    Args:
        position_ecef (np.ndarray): Receiver ECEF position [x, y, z].
        sat_positions (list): List of satellite dictionaries.
    """
    nmea_gsv = {
        'G08': {'el': 51, 'az': 302},
        'G10': {'el': 57, 'az': 23},
        'G27': {'el': 75, 'az': 232},
        'G32': {'el': 56, 'az': 133},
    }

    print("\nSatellite geometry validation vs NMEA GSV:")
    print(f"  {'Sat':<5} {'Comp El':>8} {'NMEA El':>8} {'ΔEl':>6}  "
          f"{'Comp Az':>8} {'NMEA Az':>8} {'ΔAz':>6}")
    print("  " + "-" * 62)

    for s in sat_positions:
        if s['id'] not in nmea_gsv:
            continue
        sv_xyz = np.array([s['x_sv_m'], s['y_sv_m'], s['z_sv_m']])
        el, az = ecef_to_azel(position_ecef, sv_xyz)
        ref    = nmea_gsv[s['id']]
        d_el   = el - ref['el']
        d_az   = az - ref['az']
        flag   = '✅' if abs(d_el) < 3 and abs(d_az) < 3 else '❌'
        print(f"  {s['id']:<5} {el:>8.1f} {ref['el']:>8}  {d_el:>+6.1f}  "
              f"{az:>8.1f} {ref['az']:>8}  {d_az:>+6.1f}  {flag}")
        

def ecef_vel_to_enu(vel_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """
    Convert ECEF velocity vector to local East/North/Up (ENU) components.

    Args:
        vel_ecef (np.ndarray): Velocity vector in ECEF frame [vx, vy, vz].
        lat_deg (float): Receiver latitude in degrees.
        lon_deg (float): Receiver longitude in degrees.

    Returns:
        tuple: (velocity_east, velocity_north, velocity_up) in m/s.
    """

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # Rotation matrix ECEF → ENU
    R = np.array([
        [-sin_lon,           cos_lon,            0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
    ])
    v_enu = R @ vel_ecef
    return float(v_enu[0]), float(v_enu[1]), float(v_enu[2])