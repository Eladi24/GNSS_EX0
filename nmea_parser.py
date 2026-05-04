"""
Parse GGA sentences from the Samsung GNSS log NMEA format and compare
to solver output for bias estimation.

Log format per line:
  NMEA,$GNGGA,HHMMSS.ss,DDMM.mmm,N,DDDMM.mmm,E,Q,NS,HDOP,ALT,M,GEOID,M,,*CS,unix_ms
"""
import numpy as np
from datetime import datetime, timezone


def _ddmm_to_deg(ddmm: str, hemisphere: str) -> float:
    """Convert NMEA DDMM.mmmmmm string + N/S/E/W to decimal degrees."""
    dot = ddmm.index('.')
    deg = float(ddmm[:dot - 2])
    minutes = float(ddmm[dot - 2:])
    decimal = deg + minutes / 60.0
    if hemisphere in ('S', 'W'):
        decimal = -decimal
    return decimal


def parse_gga(nmea_file: str) -> list[dict]:
    """
    Parse every GGA sentence from the log file.

    Returns list of dicts:
      unix_ms        : int    — milliseconds since Unix epoch (UTC)
      utc_time       : datetime
      lat_deg        : float  — WGS-84 latitude
      lon_deg        : float  — WGS-84 longitude
      alt_ellipsoid_m: float  — WGS-84 ellipsoid height = alt_MSL + geoid
      alt_msl_m      : float  — altitude above MSL
      geoid_m        : float  — geoid separation
      quality        : int    — fix quality (1=GPS, 4=RTK fixed, …)
      num_sats       : int
      hdop           : float
    """
    records = []
    with open(nmea_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            # Expect: NMEA, $GxGGA, time, lat, N/S, lon, E/W, quality,
            #         num_sats, hdop, alt, M, geoid, M, [dgps_age], *CS, unix_ms
            if len(parts) < 17:
                continue
            if not parts[1].endswith('GGA'):
                continue
            try:
                unix_ms  = int(parts[-1])
                quality  = int(parts[7])
                num_sats = int(parts[8])
                hdop     = float(parts[9])
                alt_msl  = float(parts[10])
                geoid    = float(parts[12])
                lat_deg  = _ddmm_to_deg(parts[3], parts[4])
                lon_deg  = _ddmm_to_deg(parts[5], parts[6])
                alt_ellipsoid = alt_msl + geoid
                utc_time = datetime.fromtimestamp(unix_ms / 1000.0, tz=timezone.utc)
                records.append({
                    'unix_ms':         unix_ms,
                    'utc_time':        utc_time,
                    'lat_deg':         lat_deg,
                    'lon_deg':         lon_deg,
                    'alt_ellipsoid_m': alt_ellipsoid,
                    'alt_msl_m':       alt_msl,
                    'geoid_m':         geoid,
                    'quality':         quality,
                    'num_sats':        num_sats,
                    'hdop':            hdop,
                })
            except (ValueError, IndexError):
                continue
    return records


def compare_to_nmea(results: list[dict], nmea_refs: list[dict],
                    max_time_diff_s: float = 0.6) -> None:
    """
    Match solver results to NMEA GGA by UTC timestamp (nearest within
    max_time_diff_s) and print error statistics.

    Errors are in ENU metres computed from the NMEA reference position.
    """
    if not nmea_refs:
        print("\nNMEA comparison: no GGA records found")
        return

    # Index NMEA records by unix_ms for fast nearest-neighbour lookup
    nmea_times = np.array([r['unix_ms'] for r in nmea_refs])

    lat_errs, lon_errs, alt_errs = [], [], []
    horiz_errs = []
    matched = 0

    METERS_PER_DEG_LAT = 111_320.0

    for res in results:
        ts = res['timestamp']
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        res_ms = int(ts.timestamp() * 1000)

        idx = int(np.argmin(np.abs(nmea_times - res_ms)))
        dt_s = abs(nmea_times[idx] - res_ms) / 1000.0
        if dt_s > max_time_diff_s:
            continue

        ref = nmea_refs[idx]
        d_lat = res['lat'] - ref['lat_deg']
        d_lon = res['lon'] - ref['lon_deg']
        d_alt = res['alt'] - ref['alt_ellipsoid_m']

        # Convert angular errors to metres
        err_n = d_lat * METERS_PER_DEG_LAT
        err_e = d_lon * METERS_PER_DEG_LAT * np.cos(np.radians(ref['lat_deg']))
        err_h = np.sqrt(err_e**2 + err_n**2)

        lat_errs.append(err_n)
        lon_errs.append(err_e)
        alt_errs.append(d_alt)
        horiz_errs.append(err_h)
        matched += 1

    if matched == 0:
        print("\nNMEA comparison: no epochs matched within time window")
        return

    lat_errs = np.array(lat_errs)
    lon_errs = np.array(lon_errs)
    alt_errs = np.array(alt_errs)
    horiz_errs = np.array(horiz_errs)

    print(f"\n{'='*60}")
    print(f"NMEA Comparison — {matched}/{len(results)} epochs matched")
    print(f"{'='*60}")
    print(f"  {'Component':<12} {'Mean bias':>10} {'Std dev':>10} {'RMSE':>10}")
    print(f"  {'-'*44}")
    print(f"  {'North (m)':<12} {np.mean(lat_errs):>+10.2f} "
          f"{np.std(lat_errs):>10.2f} {np.sqrt(np.mean(lat_errs**2)):>10.2f}")
    print(f"  {'East  (m)':<12} {np.mean(lon_errs):>+10.2f} "
          f"{np.std(lon_errs):>10.2f} {np.sqrt(np.mean(lon_errs**2)):>10.2f}")
    print(f"  {'Up    (m)':<12} {np.mean(alt_errs):>+10.2f} "
          f"{np.std(alt_errs):>10.2f} {np.sqrt(np.mean(alt_errs**2)):>10.2f}")
    print(f"  {'Horiz (m)':<12} {np.mean(horiz_errs):>10.2f} "
          f"{np.std(horiz_errs):>10.2f} {np.sqrt(np.mean(horiz_errs**2)):>10.2f}")
    print(f"{'='*60}")

    # Flag persistent biases so the user can add a correction
    bias_n = float(np.mean(lat_errs))
    bias_e = float(np.mean(lon_errs))
    bias_u = float(np.mean(alt_errs))
    if abs(bias_n) > 5 or abs(bias_e) > 5 or abs(bias_u) > 5:
        print("\n  Suggested bias correction (subtract from results):")
        print(f"    Δlat = {bias_n / METERS_PER_DEG_LAT:+.7f}°  ({bias_n:+.1f} m North)")
        print(f"    Δlon = {bias_e / (METERS_PER_DEG_LAT * np.cos(np.radians(32.17))):+.7f}°  ({bias_e:+.1f} m East)")
        print(f"    Δalt = {bias_u:+.1f} m")
