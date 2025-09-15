"""
Real-case input loaders for the optimization experiment.

Provides:
- load_geodata(): GeoData for BG_GEOID_LIST
- load_probability_from_population(): probability dict from ACS population
- load_real_odd(): Omega_dict from compute_real_odd output CSV
- load_baseline_assignment(): route-based baseline assignment (nearest-stop)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from lib.data import GeoData, RouteData
from lbbd_config import DATA_PATHS, BG_GEOID_LIST


def load_geodata() -> GeoData:
    shp = DATA_PATHS['shapefile']
    return GeoData(shp, BG_GEOID_LIST, level='block_group')


def compute_vehicle_speed_kmh(geodata: GeoData, routes_json: str = 'hct_routes.json') -> float:
    """Estimate vehicle speed in km/h from HCT routes.

    For each route, decode the polyline, project to EPSG:2163 using RouteData's
    projector, compute total path length (kilometers) and divide by total
    scheduled time (hours) from SecondsToNextStop. Return the median km/h across routes.
    """
    try:
        rd = RouteData(routes_json, geodata)
        import polyline
        from shapely.geometry import LineString

        kmh_values = []
        for route in rd.routes_info:
            poly = route.get('EncodedPolyline')
            stops = route.get('Stops', [])
            total_secs = sum(s.get('SecondsToNextStop', 0) for s in stops)
            if not poly or total_secs <= 0:
                continue
            try:
                coords = polyline.decode(poly)  # list of (lat, lon)
                line = LineString([(lng, lat) for (lat, lng) in coords])
                proj_line = rd.project_geometry(line)  # EPSG:2163 meters
                km = float(proj_line.length) / 1000.0
                hours = float(total_secs) / 3600.0
                kmh = km / hours if hours > 0 else None
                if kmh and kmh > 0:
                    kmh_values.append(kmh)
            except Exception:
                continue
        if kmh_values:
            import statistics
            return float(statistics.median(kmh_values))
    except Exception:
        pass
    # Fallback if routes not available
    return 20.0  # km/h fallback


def set_geodata_speeds(geodata: GeoData, routes_json: str = 'hct_routes.json', walk_kmh: float = 5.0) -> tuple[float, float]:
    """Set speeds on geodata consistently in km/h and return (wv_kmh, wr_kmh).

    Defaults: walking 5.0 km/h (~1.39 m/s).
    """
    wv_kmh = compute_vehicle_speed_kmh(geodata, routes_json)
    geodata.wv_kmh = wv_kmh
    geodata.wr_kmh = walk_kmh
    # Also store mph for any legacy code paths
    geodata.wv = wv_kmh / 1.60934
    geodata.wr = walk_kmh / 1.60934
    return wv_kmh, walk_kmh


def _load_population_df() -> pd.DataFrame:
    # Prefer configured B01003 CSV; fallback to provided aggregate
    path = DATA_PATHS.get('population_data')
    if path and os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        geocol = next((c for c in df.columns if c.lower().startswith('geography')), 'Geography')
        popcol = next((c for c in df.columns if c.lower().startswith('estimate total')), None)
        if popcol is None:
            for c in df.columns:
                if c.lower().startswith('estimate') and 'total' in c.lower():
                    popcol = c
                    break
        out = pd.DataFrame({'GEOID': df[geocol], 'pop': pd.to_numeric(df[popcol], errors='coerce')})
    else:
        alt = os.path.join('2023_target_blockgroup_population', '2023_target_blockgroup_population.csv')
        df = pd.read_csv(alt, dtype={'GEO_ID': str})
        out = pd.DataFrame({'GEOID': df['GEO_ID'], 'pop': pd.to_numeric(df['B02001_001E'], errors='coerce')})
    out['short_GEOID'] = out['GEOID'].str[-7:]
    return out.dropna(subset=['pop'])


def load_probability_from_population(geodata: GeoData) -> Dict[str, float]:
    df = _load_population_df()[['short_GEOID', 'pop']]
    df = df[df['short_GEOID'].isin(geodata.short_geoid_list)]
    if df['pop'].sum() <= 0:
        # fallback uniform
        n = len(geodata.short_geoid_list)
        return {sid: 1.0 / n for sid in geodata.short_geoid_list}
    df['p'] = df['pop'] / df['pop'].sum()
    return dict(zip(df['short_GEOID'], df['p']))


def load_real_odd(csv_path: str = 'data/bg_odd_features_real.csv') -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path, dtype={'short_GEOID': str})
    return {r['short_GEOID']: np.array([r['omega1'], r['omega2']], dtype=float) for _, r in df.iterrows()}


def load_baseline_assignment(geodata: GeoData, routes_json: str = 'hct_routes.json', visualize: bool = False) -> Tuple[np.ndarray, List[str]]:
    rd = RouteData(routes_json, geodata)
    assignment, centers = rd.build_assignment_matrix(visualize=visualize)
    # Convert centers (which are short_GEOID values) to strings for consistency
    centers = [str(c) for c in centers]
    return assignment, centers
