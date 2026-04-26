import rinex_parser as rp
import satellite_position as sp
import solver as slv
import numpy as np

if __name__ == "__main__":
    epochs = rp.parse_rinex_file("gnss_log_2026_03_21_17_14_34.26o")
    eph    = sp.get_ephemeris(epochs[0]['timestamp'])

    sat_positions = sp.compute_sat_position(epochs[0], eph)
    invalid = [s['id'] for s in sat_positions if not s.get('b_sv_m_valid', True)]
    if invalid:
        print(f"  Excluding (no clock correction): {', '.join(invalid)}")

    # Pass 1: GPS-only rough position for elevation mask
    gps_only  = [s for s in sat_positions if s['id'].startswith('G')]
    rough_pos, _ = slv.least_squares_position(gps_only)

    # Pass 2: all constellations above elevation mask
    if rough_pos is not None:
        filtered = sp.filter_by_elevation(sat_positions, rough_pos)
        pos, dt  = slv.least_squares_position(filtered)
    else:
        pos, dt  = slv.least_squares_position(sat_positions)

    if pos is None:
        print("  Solution failed")
        exit(1)

    lat, lon, alt = slv.ecef_to_geodetic(*pos)

    print(f"\nReceiver ECEF position:")
    print(f"  X = {pos[0]:+.3f} m")
    print(f"  Y = {pos[1]:+.3f} m")
    print(f"  Z = {pos[2]:+.3f} m")
    print(f"  dT = {dt:.9f} s")
    print(f"\nGeodetic position:")
    print(f"  Lat = {lat:.6f}°")
    print(f"  Lon = {lon:.6f}°")
    print(f"  Alt = {alt:.1f} m")
    print(f"\nGoogle Maps link:")
    print(f"  https://maps.google.com/?q={lat:.6f},{lon:.6f}")

    slv.validate_with_gsv(pos, sat_positions)
    vel, f_dot = slv.least_squares_velocity(sat_positions, pos)

    if vel is not None:
        speed_ms  = np.linalg.norm(vel)
        speed_kmh = speed_ms * 3.6
        print(f"\nReceiver velocity (ECEF):")
        print(f"  Vx = {vel[0]:+.3f} m/s")
        print(f"  Vy = {vel[1]:+.3f} m/s")
        print(f"  Vz = {vel[2]:+.3f} m/s")
        print(f"  Speed = {speed_ms:.2f} m/s  ({speed_kmh:.1f} km/h)")
        print(f"  Clock drift = {f_dot:.4f} m/s  "
            f"({f_dot/slv.SPEED_OF_LIGHT*1e9:.3f} ns/s)")