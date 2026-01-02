from skyfield.api import load

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
MAX_NODES = 500
ISL_RANGE_KM = 2000
DURATION_MINUTES = 95
TIME_STEP_MIN = 1


def fetch_satellite_data():
    print("Fetching TLE data from CelesTrak...")
    satellites = load.tle_file(TLE_URL)
    print(f"Total satellites found: {len(satellites)}")
    subset = satellites[:MAX_NODES]
    print(f"Selected {len(subset)} satellites for analysis")
    return subset


fetch_satellite_data()
