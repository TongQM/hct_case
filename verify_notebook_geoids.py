#!/usr/bin/env python3
"""
Verify that all block groups from data_process.ipynb are available in the shapefile
"""

import geopandas as gpd
import os
from lbbd_config import BG_GEOID_LIST

def verify_notebook_geoids():
    """Verify all GEOIDs from the notebook are available"""
    
    bgfile_path = "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    
    if not os.path.exists(bgfile_path):
        print(f"ERROR: Shapefile not found: {bgfile_path}")
        return False
    
    try:
        print("Reading shapefile...")
        gdf = gpd.read_file(bgfile_path)
        print(f"✓ Shapefile loaded with {len(gdf)} block groups")
        
        # Create short GEOID column like in the notebook
        gdf["short_GEOID"] = gdf["GEOID"].str[-7:]
        
        print(f"\nChecking {len(BG_GEOID_LIST)} GEOIDs from notebook...")
        
        # Convert to short GEOIDs
        short_geoids = [geoid[-7:] for geoid in BG_GEOID_LIST]
        
        # Check availability
        available = []
        missing = []
        
        for i, (full_geoid, short_geoid) in enumerate(zip(BG_GEOID_LIST, short_geoids)):
            if short_geoid in gdf["short_GEOID"].values:
                available.append((full_geoid, short_geoid))
            else:
                missing.append((full_geoid, short_geoid))
        
        print(f"\nResults:")
        print(f"✓ Available: {len(available)}/{len(BG_GEOID_LIST)} ({len(available)/len(BG_GEOID_LIST)*100:.1f}%)")
        print(f"✗ Missing: {len(missing)} ({len(missing)/len(BG_GEOID_LIST)*100:.1f}%)")
        
        if missing:
            print(f"\nMissing GEOIDs:")
            for full_geoid, short_geoid in missing[:10]:  # Show first 10
                print(f"  {full_geoid} -> {short_geoid}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        
        if available:
            print(f"\nFirst 10 available GEOIDs:")
            for full_geoid, short_geoid in available[:10]:
                print(f"  ✓ {full_geoid} -> {short_geoid}")
        
        # Test GeoData initialization with available GEOIDs only
        if available:
            print(f"\nTesting GeoData initialization with available GEOIDs...")
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
                from lib import GeoData
                
                available_full_geoids = [item[0] for item in available]
                bg_geodata = GeoData(bgfile_path, available_full_geoids, level='block_group')
                print(f"✓ GeoData initialized successfully with {len(bg_geodata.short_geoid_list)} block groups")
                
                # Test arc structures
                arc_list = bg_geodata.get_arc_list()
                print(f"✓ Arc structures created: {len(arc_list)} directed arcs")
                
                return True
                
            except Exception as e:
                print(f"✗ GeoData initialization failed: {e}")
                return False
        else:
            print(f"✗ No available GEOIDs to test")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = verify_notebook_geoids()
    sys.exit(0 if success else 1) 