#!/usr/bin/env python3
"""
Debug GEOID format in the shapefile
"""

import geopandas as gpd
import os

def debug_geoid_format():
    """Debug the GEOID format in the shapefile"""
    
    bgfile_path = "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    
    if not os.path.exists(bgfile_path):
        print(f"ERROR: Shapefile not found: {bgfile_path}")
        return
    
    try:
        print("Reading shapefile...")
        gdf = gpd.read_file(bgfile_path)
        print(f"✓ Shapefile loaded with {len(gdf)} block groups")
        
        print(f"\nColumns in shapefile:")
        for col in gdf.columns:
            print(f"  {col}")
        
        print(f"\nFirst 10 full GEOIDs:")
        for i, geoid in enumerate(gdf["GEOID"].head(10)):
            print(f"  {i+1}: {geoid} (length: {len(geoid)})")
        
        # Check the format of one of our requested GEOIDs
        requested_full = "1500000US420034980001"
        print(f"\nAnalyzing requested GEOID: {requested_full}")
        print(f"  Length: {len(requested_full)}")
        print(f"  Last 7 chars: {requested_full[-7:]}")
        print(f"  Last 12 chars: {requested_full[-12:]}")
        
        # Check if any GEOID contains our pattern
        pattern = "4980001"
        matches = gdf[gdf["GEOID"].str.contains(pattern, na=False)]
        print(f"\nSearching for pattern '{pattern}':")
        print(f"  Found {len(matches)} matches")
        for _, row in matches.head(5).iterrows():
            print(f"    {row['GEOID']}")
        
        # Try different extraction methods
        print(f"\nTrying different GEOID extraction methods:")
        sample_geoid = gdf["GEOID"].iloc[0]
        print(f"  Sample GEOID: {sample_geoid}")
        print(f"  Last 7 chars: {sample_geoid[-7:]}")
        print(f"  Last 12 chars: {sample_geoid[-12:]}")
        print(f"  After removing prefix: {sample_geoid.replace('1500000US42003', '')}")
        
        # Check if the requested GEOIDs exist with different formatting
        requested_geoids = [
            "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
            "1500000US420034993002", "1500000US420034994001"
        ]
        
        print(f"\nChecking requested GEOIDs:")
        for req_geoid in requested_geoids:
            # Try exact match
            exact_match = gdf[gdf["GEOID"] == req_geoid]
            print(f"  {req_geoid}: exact match = {len(exact_match)}")
            
            # Try without the prefix
            without_prefix = req_geoid.replace("1500000US", "")
            prefix_match = gdf[gdf["GEOID"] == without_prefix]
            print(f"    {without_prefix}: prefix removed = {len(prefix_match)}")
            
            # Try pattern matching
            pattern = req_geoid[-7:]
            pattern_match = gdf[gdf["GEOID"].str.endswith(pattern)]
            print(f"    ending with {pattern}: pattern match = {len(pattern_match)}")
        
        # Show some actual GEOIDs that might be similar
        print(f"\nSample of actual GEOIDs that might be in the area:")
        area_geoids = gdf[gdf["GEOID"].str.contains("498", na=False)]["GEOID"].head(10)
        for geoid in area_geoids:
            print(f"  {geoid}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_geoid_format() 