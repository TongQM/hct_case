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

