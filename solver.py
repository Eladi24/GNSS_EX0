import numpy as np

SPEED_OF_LIGHT     = 299_792_458.0
CONSTELLATION_ORDER = ['G', 'R', 'E', 'C']  # GPS is reference clock


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
    # TODO: handle multiple constellations with ISBs, right now suppose GPS-only
    valid = [s for s in sat_positions
             if s['id'].startswith('G')
             and s.get('doppler') is not None
             and s.get('doppler_lambda') is not None
             and s.get('b_sv_m_valid', True)]
    if len(valid) < 4:
        print(f"  Not enough satellites for velocity solution "
              f"(have {len(valid)}, need 4)")
        for s in sat_positions:
            if not s['id'].startswith('G'):
                continue
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

    # One-step linear solve
    result = np.linalg.lstsq(H, obs, rcond=None)[0]
    vel = result[:3]
    f_dot = result[3]
    return vel, f_dot

def least_squares_position(sat_positions: list) -> tuple[np.ndarray, float]:
    """
    Multi-constellation Gauss-Newton least squares position solver.

    GPS is the reference clock. Each additional constellation present
    in sat_positions gets its own Inter-System Bias (ISB) column in H,
    estimated jointly with position and receiver clock.

    Parameters
    ----------
    sat_positions : list of dicts
        Output from compute_sat_position() or filter_by_elevation().
        Each dict must have: id, corrected_pr, x_sv_m, y_sv_m, z_sv_m

    Returns
    -------
    pos : np.ndarray shape (3,)
        Receiver ECEF position [x, y, z] in metres
    dt : float
        Receiver GPS clock bias in seconds
    """
    supported = [s for s in sat_positions
                 if s['id'][0] in CONSTELLATION_ORDER
                 and s.get('b_sv_m_valid', True)]
    if not supported:
        return None, None

    # Identify extra constellations actually present (GPS is reference)
    present_sys = [sys for sys in CONSTELLATION_ORDER[1:]
                   if any(s['id'].startswith(sys) for s in supported)]

    n_unknowns = 4 + len(present_sys)  # x, y, z, cdt_GPS, [ISBs...]

    if len(supported) < n_unknowns:
        print(f"  Not enough satellites ({len(supported)}) for "
              f"{n_unknowns} unknowns — falling back to GPS only")
        supported  = [s for s in sat_positions if s['id'].startswith('G')]
        present_sys = []
        n_unknowns  = 4
        if len(supported) < 4:
            print("  Not enough GPS satellites for solution")
            return None, None

    sys_ids      = [s['id'][0] for s in supported]
    sat_xyz      = np.array([[s['x_sv_m'], s['y_sv_m'], s['z_sv_m']]
                              for s in supported])
    pseudoranges = np.array([s['corrected_pr'] for s in supported])

    print(f"  Using {len(supported)} satellites "
          f"({', '.join(['GPS'] + present_sys)})")

    # Initial state: surface of Earth at origin, zero clock biases
    state    = np.zeros(n_unknowns)
    state[2] = 6_371_000.0

    for iteration in range(10):
        x, y, z  = state[0], state[1], state[2]
        cdt_gps  = state[3]

        diff   = sat_xyz - np.array([x, y, z])
        ranges = np.linalg.norm(diff, axis=1)

        # Clock bias per satellite: GPS baseline + ISB for other systems
        clock_bias = np.full(len(supported), cdt_gps)
        for k, sys in enumerate(present_sys):
            for i, sid in enumerate(sys_ids):
                if sid == sys:
                    clock_bias[i] = cdt_gps + state[4 + k]

        residuals = pseudoranges - (ranges + clock_bias)
        unit_vecs = -diff / ranges[:, np.newaxis]

        # H = [direction cosines | GPS clock | ISB columns]
        H = np.zeros((len(supported), n_unknowns))
        H[:, :3] = unit_vecs
        H[:, 3]  = 1.0
        for k, sys in enumerate(present_sys):
            for i, sid in enumerate(sys_ids):
                if sid == sys:
                    H[i, 4 + k] = 1.0

        delta  = np.linalg.lstsq(H, residuals, rcond=None)[0]
        state += delta

        if np.linalg.norm(delta[:3]) < 1e-4:
            print(f"  Converged in {iteration + 1} iterations")
            break

    pos = state[:3]
    dt  = state[3] / SPEED_OF_LIGHT

    if present_sys:
        print("  Inter-system biases:")
        for k, sys in enumerate(present_sys):
            isb_m = state[4 + k]
            print(f"    GPS→{sys}: {isb_m/1000:.3f} km  "
                  f"({isb_m / SPEED_OF_LIGHT * 1e9:.1f} ns)")

    return pos, dt


def ecef_to_geodetic(x, y, z) -> tuple[float, float, float]:
    """
    Convert ECEF (metres) to geodetic (lat, lon, alt).
    Uses Bowring's iterative method with WGS-84 ellipsoid.
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
    """Compute elevation and azimuth (degrees) from receiver to satellite."""
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
    """Compare computed satellite geometry against NMEA GSV ground truth."""
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