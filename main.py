"""
Main entry point for the GNSS Processing Pipeline.
Orchestrates reading RINEX files, downloading ephemeris data, 
computing satellite positions, and estimating receiver position/velocity.
"""

import numpy as np
import rinex_parser as rp
import satellite_position as sp
import solver as slv
import output as out
import corrections as cor
import nmea_parser
from datetime import timezone

# ── Universal physical constraints (valid anywhere on Earth) ────────────────
ALTITUDE_MIN_M   = -500.0     # Dead Sea is the deepest land point (-430 m)
ALTITUDE_MAX_M   = 15_000.0   # above Everest (8849 m); covers commercial aircraft
MAX_SPEED_MS     = 300.0      # m/s ≈ Mach 1; catches teleportation-like jumps
MAX_RESIDUAL_RMS = 5000.0     # metres; smartphone DCBs cause ~1000 m per-satellite residuals


def solution_is_valid(pos_ecef: np.ndarray,
                      alt: float,
                      residuals: np.ndarray,
                      prev_pos: np.ndarray = None,
                      dt_seconds: float    = None) -> tuple[bool, str]:
    """
    Universal validity check for a computed GNSS solution, making no geographic assumptions.
    Ensures altitude, residual RMS, and speed are physically plausible.

    Args:
        pos_ecef (np.ndarray): Computed ECEF position [x, y, z] in metres.
        alt (float): Computed geodetic altitude in metres.
        residuals (np.ndarray): Post-fit residuals from the least-squares solver.
        prev_pos (np.ndarray, optional): Receiver ECEF position from the previous epoch.
        dt_seconds (float, optional): Time difference since the previous epoch in seconds.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating validity, 
                          and a string explaining the reason (or "ok" if valid).
    """
    # Gate 1: altitude must be physically possible on Earth
    if not (ALTITUDE_MIN_M < alt < ALTITUDE_MAX_M):
        return False, f"alt={alt:.0f} m outside [{ALTITUDE_MIN_M:.0f}, {ALTITUDE_MAX_M:.0f}] m"

    # Gate 2: post-fit residual RMS — catches wrong LS local minima
    if residuals is not None and len(residuals) > 0:
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        if rms > MAX_RESIDUAL_RMS:
            return False, f"residual RMS={rms:.1f} m > {MAX_RESIDUAL_RMS} m"

    # Gate 3: velocity consistency between consecutive epochs
    if prev_pos is not None and dt_seconds is not None and dt_seconds > 0:
        speed = float(np.linalg.norm(pos_ecef - prev_pos) / dt_seconds)
        if speed > MAX_SPEED_MS:
            return False, f"implied speed={speed:.0f} m/s > {MAX_SPEED_MS} m/s"

    return True, "ok"


if __name__ == "__main__":
    # 1. Parse observation data from RINEX file
    epochs = rp.parse_rinex_file("rinex_logs/gnss_log_2026_03_22_08_44_21.26o")
    
    # 2. Download broadcast ephemeris (nav data) for the relevant date
    eph, file_paths = sp.get_ephemeris(epochs[0]['timestamp'])

    # 3. Load ionospheric Klobuchar coefficients from the nav file
    cor.load_klobuchar_from_nav(file_paths[0])

    # 4. Crop ephemeris down to just the time window and satellites we need
    t0    = epochs[0]['timestamp'].replace(tzinfo=timezone.utc)
    ephem = sp.crop_ephemeris(epochs, sp.dt_to_gps_millis(t0))
    if ephem is None:
        print("Failed to crop ephemeris — aborting")
        exit(1)
        
    # Diagnostic: Check the clock bias for the first epoch
    print("\n=== b_sv_m diagnostic (epoch 1, all constellations) ===")
    test_epoch = epochs[0]
    test_sats  = sp.compute_sat_position(test_epoch, ephem, rx_pos=None)
    for s in sorted(test_sats, key=lambda x: x['id']):
        raw_pr  = s['pseudorange']
        b       = s['b_sv_m']
        corr_pr = s['corrected_pr']
        el      = s.get('elevation_deg')
        el_str  = f"{el:5.1f}°" if el is not None else "  N/A"
        print(f"  {s['id']}:  raw_pr={raw_pr/1e6:.3f}Mm  "
            f"b_sv_m={b:+.1f}m  "
            f"corrected_pr={corr_pr/1e6:.3f}Mm  "
            f"el={el_str}  valid={s['b_sv_m_valid']}")
    print("===\n")

    # Initialize tracking variables for the epoch loop
    results   = []
    last_pos  = None
    last_time = None
    last_alt  = None

    # Session-level satellite blacklisting: satellites that are repeatedly
    # flagged as outliers get excluded for the rest of the session.
    outlier_history  = {}   # sat_id → count of epochs where it was an outlier
    session_blacklist: set[str] = set()
    BLACKLIST_THRESHOLD = 6

    # 5. Process each epoch sequentially
    for i, epoch in enumerate(epochs):
        print(f"\n--- Epoch {i+1}/{len(epochs)}  {epoch['timestamp']} ---")

        # Compute satellite positions, clock biases, and atmospheric corrections
        sat_positions = sp.compute_sat_position(epoch, ephem, rx_pos=last_pos)
        if not sat_positions:
            print("  No satellites — skipping")
            continue

        # Drop session-blacklisted satellites before doing any geometry work.
        if session_blacklist:
            sat_positions = [s for s in sat_positions
                             if s['id'] not in session_blacklist]

        # Apply elevation mask: use last good position as reference, or fall
        # back to constellation centroid so first epoch is also filtered.
        rough_pos = last_pos
        if rough_pos is None:
            sat_xyz   = np.array([[s['x_sv_m'], s['y_sv_m'], s['z_sv_m']]
                                   for s in sat_positions])
            rough_pos = slv._guess_initial_position(sat_xyz)
            
        # Apply elevation mask to filter out low-elevation satellites
        sat_positions = sp.filter_by_elevation(sat_positions, rough_pos,
                                               min_elevation_deg=15.0)
        if not sat_positions:
            print("  No satellites above elevation mask — skipping")
            continue

        h_constraint = last_alt if last_alt is not None else 50.0
        # Compute receiver position and clock bias using least squares
        pos, dt, residuals, outlier_ids = slv.least_squares_position(
            sat_positions,
            initial_pos=last_pos,
            height_constraint_m=h_constraint,
        )

        # Update outlier history and grow blacklist regardless of solution validity.
        for sid in outlier_ids:
            outlier_history[sid] = outlier_history.get(sid, 0) + 1
            if (outlier_history[sid] >= BLACKLIST_THRESHOLD
                    and sid not in session_blacklist):
                print(f"  Blacklisting {sid} — outlier in "
                      f"{outlier_history[sid]} epochs")
                session_blacklist.add(sid)

        if pos is None:
            print("  Position solution failed — skipping")
            continue

        # Convert ECEF position to geodetic (Latitude, Longitude, Altitude)
        lat, lon, alt = slv.ecef_to_geodetic(*pos)

        curr_time = epoch['timestamp']
        dt_sec    = ((curr_time - last_time).total_seconds()
                     if last_time is not None else None)

        # Validate the computed solution against physical constraints
        valid, reason = solution_is_valid(pos, alt, residuals, last_pos, dt_sec)
        if not valid:
            print(f"  Rejected ({reason}): lat={lat:.3f}° lon={lon:.3f}° alt={alt:.0f}m")
            continue

        last_pos  = pos
        last_time = curr_time
        # Only carry forward an altitude that is plausible for a ground receiver;
        # a bad multi-constellation solution can give km-scale altitudes that
        # would then corrupt the height constraint for every subsequent epoch.
        if ALTITUDE_MIN_M < alt < 500.0:
            last_alt = alt

        # Compute receiver velocity and clock drift from Doppler observables
        vel, f_dot = slv.least_squares_velocity(sat_positions, pos)

        if vel is not None:
            v_east, v_north, v_up = slv.ecef_vel_to_enu(vel, lat, lon)
            speed_ms = np.linalg.norm(vel)
            print(f"  Lat={lat:.6f}°  Lon={lon:.6f}°  Alt={alt:.1f}m  "
                  f"Speed={speed_ms:.2f} m/s")
        else:
            v_east = v_north = v_up = speed_ms = None
            print(f"  Lat={lat:.6f}°  Lon={lon:.6f}°  Alt={alt:.1f}m")

        rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals is not None else None
        # Save the result for this epoch
        results.append({
            'timestamp': epoch['timestamp'],
            'lat': lat, 'lon': lon, 'alt': alt,
            'v_east': v_east, 'v_north': v_north, 'v_up': v_up,
            'speed_ms': speed_ms, 'dt': dt,
            'num_sats': len(sat_positions),
            'residual_rms': rms,
        })

    print(f"\nSolved {len(results)}/{len(epochs)} epochs")
    
    # 6. Export results to CSV and KML formats
    out.write_csv(results, "output/path_2026_03_22_08_44_21.csv")
    out.write_kml(results, "output/path_2026_03_22_08_44_21.kml")

    # 7. Compare computed path to NMEA ground truth reference
    nmea_refs = nmea_parser.parse_gga("rinex_logs/gnss_log_2026_03_22_08_44_20.nmea")
    nmea_parser.compare_to_nmea(results, nmea_refs)