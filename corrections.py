import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
GPS_L1_FREQ    = 1_575_420_000.0

KLOBUCHAR_ALPHA = None
KLOBUCHAR_BETA  = None

def load_klobuchar_from_nav(nav_file_path: str):
    global KLOBUCHAR_ALPHA, KLOBUCHAR_BETA
    alpha, beta = [], []
    try:
        with open(nav_file_path, 'r') as f:
            for line in f:
                if 'IONOSPHERIC CORR' in line:
                    label = line[0:4].strip()
                    vals  = []
                    for j in range(4):
                        raw = line[5 + j*12 : 17 + j*12].strip()
                        if raw:
                            vals.append(float(raw.replace('D','e').replace('d','e')))
                    if label == 'GPSA' and len(vals) == 4:
                        alpha = vals
                    elif label == 'GPSB' and len(vals) == 4:
                        beta = vals
                if 'END OF HEADER' in line:
                    break
        if len(alpha) == 4 and len(beta) == 4:
            KLOBUCHAR_ALPHA = alpha
            KLOBUCHAR_BETA  = beta
            print(f"  Klobuchar coefficients loaded ✅")
            print(f"    alpha={alpha}")
            print(f"    beta={beta}")
        else:
            print(f"  Klobuchar not found in nav file — iono correction disabled")
    except Exception as e:
        print(f"  Warning: could not load Klobuchar: {e}")

def klobuchar_iono(gps_millis: float, sat_el_rad: float, sat_az_rad: float,
                   rx_lat_rad: float, rx_lon_rad: float,
                   alpha: list, beta: list) -> float:
    """
    Klobuchar single-frequency ionospheric correction.
    Returns correction in metres to ADD to corrected_pr.
    alpha, beta: 4-element lists from RINEX nav header ION ALPHA / ION BETA.
    """
    psi = 0.0137 / (sat_el_rad / np.pi + 0.11) - 0.022
    lat_i = rx_lat_rad / np.pi + psi * np.cos(sat_az_rad)
    lat_i = np.clip(lat_i, -0.416, 0.416)
    lon_i = rx_lon_rad / np.pi + psi * np.sin(sat_az_rad) / np.cos(lat_i * np.pi)
    lat_m = lat_i + 0.064 * np.cos((sat_az_rad - 1.617) * np.pi)

    t = 4.32e4 * lon_i +(gps_millis / 1000) % 86400
    t = t % 86400

    per = sum(beta[n] * (lat_m ** n) for n in range(4))
    per = max(per, 72000.0)

    amp = sum(alpha[n] * (lat_m ** n) for n in range(4))
    amp = max(amp, 0.0)

    x = 2 * np.pi * (t - 50400) / per
    if abs(x) < 1.57:
        iono_s = 5e-9 + amp * (1 - x**2 / 2 + x**4 / 24)
    else:
        iono_s = 5e-9
    
    # Obliquity factor
    f = 1.0 + 16.0 * (0.53 - sat_el_rad / np.pi) ** 3
    return -SPEED_OF_LIGHT * iono_s * f # negative = shorten pseudorange

def saastamoinen_tropo(sat_el_rad: float, alt_m: float = 0.0) -> float:
    """
    Simplified Saastamoinen tropospheric correction.
    Returns correction in metres to ADD to corrected_pr (always negative).
    alt_m: receiver altitude in metres (use 0 if unknown).
    """

    P = 1013.25 * (1 - 2.2557e-5 * alt_m) ** 5.2568 # pressure hPa
    T = 15.0- 6.5e-3 * alt_m + 273.15 # temperature K
    e = 0.5 * np.exp(17.27 * (T - 273.15) / (T - 36.85)) # humidity hPa

    z = np.pi / 2 - sat_el_rad # zenith angle
    # 1.156 is the standard Saastamoinen correction term B for h=0. 28.0 is an error.
    delay = (0.002277 / np.cos(z)) * (P + (1255 / T + 0.05) * e - 1.156 * np.tan(z)**2)
    return -delay # negative = shorten pseudorange