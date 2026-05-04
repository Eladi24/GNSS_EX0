import csv
from datetime import timezone
import simplekml

def write_csv(results: list, path: str):
    """
    Write one row per epoch to a CSV file.
    Columns: UTC_time, lat, lon, alt, v_east, v_north, v_up, speed_ms, num_sats
    """
    fieldnames = [
        'UTC_time', 'lat', 'lon', 'alt_m',
        'v_east_ms', 'v_north_ms', 'v_up_ms', 'speed_ms', 'num_sats'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ts = r['timestamp']
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            writer.writerow({
                'UTC_time':   ts.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                'lat':        f"{r['lat']:.8f}",
                'lon':        f"{r['lon']:.8f}",
                'alt_m':      f"{r['alt']:.2f}",
                'v_east_ms':  f"{r['v_east']:.4f}"  if r['v_east']  is not None else '',
                'v_north_ms': f"{r['v_north']:.4f}" if r['v_north'] is not None else '',
                'v_up_ms':    f"{r['v_up']:.4f}"    if r['v_up']    is not None else '',
                'speed_ms':   f"{r['speed_ms']:.4f}" if r['speed_ms'] is not None else '',
                'num_sats':   r['num_sats'],
            })
    print(f"  CSV:  {len(results)} rows → {path}")

def write_kml(results: list, path: str):
    """
    Write a KML file with a LineString track and one placemark per epoch.
    """
    kml = simplekml.Kml()
    track = kml.newlinestring(name="GNSS Track")

    # LineString coordinates: (lon, lat, alt) — KML convention
    track.coords = [(r['lon'], r['lat'], r['alt']) for r in results]
    track.altitudemode = simplekml.AltitudeMode.absolute
    track.style.linestyle.color = simplekml.Color.cyan
    track.style.linestyle.width = 3

    # One placemark per epoch
    folder = kml.newfolder(name="Epochs")
    for r in results:
        ts = r['timestamp'].strftime('%H:%M:%S')
        spd = f"{r['speed_ms']:.1f} m/s" if r['speed_ms'] is not None else "—"
        pnt = folder.newpoint(
            name        = ts,
            coords      = [(r['lon'], r['lat'], r['alt'])],
            description = (f"Alt: {r['alt']:.1f} m\n"
                           f"Speed: {spd}\n"
                           f"Sats: {r['num_sats']}")
        )
        pnt.altitudemode = simplekml.AltitudeMode.absolute
        
    kml.save(path)
    print(f"  KML: {len(results)} placemarks → {path}")
