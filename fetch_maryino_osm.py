"""Download a tiled OSM snapshot for Maryino from the OSM 0.6 API."""

import time
import urllib.error
import urllib.request
from pathlib import Path


OUT = Path(__file__).resolve().parent / "data" / "raw_osm"
BASE = "https://api.openstreetmap.org/api/0.6"
BBOX = (37.6929, 55.6394, 37.7881, 55.6689)
NX, NY = 4, 3


def fetch(url, path):
    if path.exists() and path.stat().st_size > 1000:
        print(f"Already present: {path.name}", flush=True)
        return
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Maryino-POI-research/0.1 (OpenStreetMap API)"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if not data.startswith(b"<?xml"):
                raise ValueError(f"Unexpected response from {url}: {data[:100]!r}")
            path.write_bytes(data)
            print(f"Downloaded: {path.name} ({len(data)} bytes)", flush=True)
            time.sleep(1)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise
            print(f"Retrying {path.name}: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fetch(f"{BASE}/relation/455222/full", OUT / "boundary.osm")
    west, south, east, north = BBOX
    for iy in range(NY):
        for ix in range(NX):
            left = west + (east - west) * ix / NX
            right = west + (east - west) * (ix + 1) / NX
            bottom = south + (north - south) * iy / NY
            top = south + (north - south) * (iy + 1) / NY
            bbox = ",".join(f"{v:.7f}" for v in (left, bottom, right, top))
            fetch(f"{BASE}/map?bbox={bbox}", OUT / f"tile_{iy}_{ix}.osm")


if __name__ == "__main__":
    main()
