import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from libpysal.weights import Queen
import polyline
import json
from shapely.geometry import LineString, Point
from shapely.ops import transform
from pyproj import Transformer

from lib.data import GeoData, RouteData

'''
Compare the costs incurred by different entities using different partitions plus operations.

We will compare the costs on the riders and the costs on the provider.

For fixed-route mode, the costs on the riders include
    1) Walk time, expected walk distance from home to the nearest stop divided by the average walk speed 1.4 m/s
    2) Wait time, expected wait time at the stop until the shuttle arrives
    3) Transit time, expected transit time from the stop to the destination on the shuttle

For TSP mode, the costs on the riders include
    1) Wait time, expected wait time at home until the shuttle arrives
    2) Transit time, expected transit time from home to the destination on the shuttle

For the provider, the cost is
    1) Travel cost, the long-run average cost of the shuttle operting on a district

Besides the costs on the riders and the provider, we also want to capture the ridership based on the service and region design.

We will compare the different costs for different service designs.

    1) Original partition with fixed-route mode
    2) Optimal partition with TSP mode
    3) Optimal partition with TSP mode with max clearing interval constraints
'''


class Evaluate:

    def __init__(self, routedata: RouteData, geodata: GeoData):
        
        assert isinstance(routedata, RouteData), "routedata must be of type RouteData"
        assert isinstance(geodata, GeoData), "geodata must be of type GeoData"
        assert routedata.geodata == geodata, "routedata and geodata must be the same instance"

        self.routedata = routedata
        self.geodata = geodata
        self.level = geodata.level
        self.short_geoid_list = geodata.short_geoid_list


    def evaluate_fixed_route(self, prob_dict):
        """
        Evalaute the costs incurred by riders and the provider using fixed-route mode.
        For riders, the costs include
            1) Walk time, expected walk distance from home to the nearest stop divided by the average walk speed 1.4 m/s
            2) Wait time, expected wait time at the stop until the shuttle arrives
            3) Transit time, expected transit time from the stop to the destination on the shuttle
        For the provider, the cost is
            1) Travel cost, the long-run average cost of the shuttle operting on a district
        """
        """
        Calculate walk, wait, and transit times for fixed-route mode.
        Returns a dict with detailed and expected metrics.
        """
        routes_info = self.routedata.routes_info
        # 1) Route lengths and projected stops
        route_lengths = {}
        route_projected_stops = {}
        for route in routes_info:
            name = route.get("Description", f"Route {route.get('RouteID')}")
            poly = route.get("EncodedPolyline")
            if poly:
                coords = polyline.decode(poly)
                line = LineString([(lng, lat) for lat, lng in coords])
                proj_line = self.routedata.project_geometry(line)
                route_lengths[name] = proj_line.length
            stops = route.get("Stops", [])
            proj_stops = [self.routedata.project_geometry(Point(s['Longitude'], s['Latitude'])) for s in stops]
            route_projected_stops[name] = proj_stops

        # 2) Travel times from stops data
        travel_times = {}
        for route in routes_info:
            name = route.get("Description", f"Route {route.get('RouteID')}")
            total = sum(s.get('SecondsToNextStop', 0) for s in route.get('Stops', []))
            travel_times[name] = total

        # 3) Walk distances weighted by population
        prob = prob_dict  # mapping node idx -> probability mass
        walk_distance = {r: 0.0 for r in route_projected_stops}
        for idx, row in self.geodata.gdf.iterrows():
            district = row['district']
            if district:
                stops = route_projected_stops[district]
                if stops:
                    dmin = min(row.geometry.centroid.distance(st) for st in stops)
                else:
                    dmin = 0.0
                walk_distance[district] += dmin * prob.get(idx, 0)

        expected_walk_distance = sum(walk_distance.values())
        walking_speed = 1.4  # m/s
        expected_walk_time = expected_walk_distance / walking_speed
        walk_time_detail = {r: walk_distance[r]/walking_speed for r in walk_distance}

        # 4) Wait time: assume uniform arrival, average wait = half cycle time
        wait_time = {r: travel_times[r]/2.0 for r in travel_times}
        # group-level probability for each route
        district_probs = {
            r: sum(prob.get(idx, 0) for idx in self.geodata.gdf[self.geodata.gdf['district']==r].index)
            for r in wait_time
        }
        expected_wait_time = sum(wait_time[r]*district_probs[r] for r in wait_time)

        # 5) Transit time: assume average ride = half total travel time
        transit_time = {r: travel_times[r]/2.0 for r in travel_times}
        expected_transit_time = sum(transit_time[r]*district_probs[r] for r in transit_time)

        return {
            'route_lengths': route_lengths,
            'travel_times': travel_times,
            'walk_time_detail': walk_time_detail,
            'expected_walk_time': expected_walk_time,
            'wait_time_detail': wait_time,
            'expected_wait_time': expected_wait_time,
            'transit_time_detail': transit_time,
            'expected_transit_time': expected_transit_time
        }

        

    def evaluate(self, node_assignment, center_list, prob_dict, mode="fixed"):
        """
        Evaluate the costs incurred by different entities using different partitions plus operations.
        We will compare the costs on the riders and the costs on the provider.
        We will compare the different costs for different service designs.
        1. Original partition with fixed-route mode
        2. Optimal partition with TSP mode
        3. Optimal partition with max clearing interval constraints
        """

        assert mode in ["fixed", "tsp"], "mode must be either 'fixed' or 'tsp'"
        assert node_assignment is not None, "node_assignment must not be None"
        assert center_list is not None, "center_list must not be None"

        if mode == "fixed":
            results_dict = self.evaluate_fixed_route(prob_dict)
            return results_dict
        elif mode == "tsp":
            pass
        





        