import geopandas as gpd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
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

    def __init__(self, routedata: RouteData, geodata: GeoData, epsilon=1e-3):
        '''
        Initialize the Evaluate class with route data and geo data.
        :param routedata: RouteData object
        :param geodata: GeoData object
        :param epsilon: Radius for Wasserstein distance
        '''
        
        assert isinstance(routedata, RouteData), "routedata must be of type RouteData"
        assert isinstance(geodata, GeoData), "geodata must be of type GeoData"
        assert routedata.geodata == geodata, "routedata and geodata must be the same instance"

        self.routedata = routedata
        self.geodata = geodata
        self.level = geodata.level
        self.short_geoid_list = geodata.short_geoid_list
        self.epsilon = epsilon


    def _get_fixed_route_worst_distribution(self, prob_dict):
        """
        Get the worst-case distribution for fixed-route mode.
        Since for the fixed-route mode, the travel distance is sunk, the operating cost on the provider is fixed.
        The costs we need to consider are the costs on the riders, which include wait time, transit time, and walk time.
        Given that wait time and trainsit time are both solely dependent on the routes, they are also sunk.
        Therefore, the only cost we need to consider is the walk time, which is dependent on the distribution of the riders.
        The worst-case distribution is the one that maximizes the walk time, within the Wasserstein ball centered at the true distribution.

        :param prob_dict: dict
            A dictionary that maps the node index to the underlying true probability mass.
            The keys are the node indices, and the values are the probability masses.
        """
        nearest_stop = self.routedata.find_nearest_stops()
        node_list = self.geodata.short_geoid_list

        model = gp.Model("fixed_route_worst_distribution")

        prob_mass = model.addVars(node_list, lb=0.0, vtype=GRB.CONTINUOUS, name="prob_mass")
        transport_plan = model.addVars(node_list, node_list, lb=0.0, vtype=GRB.CONTINUOUS, name="transport_plan")
        # Objective: maximize the total walk time
        walk_time_dict = {node: nearest_stop[node]["distance"] / 1.4 for node in node_list}
        model.setObjective(gp.quicksum(prob_mass[node] * walk_time_dict[node] for node in node_list), GRB.MAXIMIZE)

        # Constraints
        model.addConstr(prob_mass.sum() == 1, "total_prob_mass")
        model.addConstrs(gp.quicksum(transport_plan[node1, node2] for node2 in node_list) == prob_dict[node1] for node1 in node_list)
        model.addConstrs(gp.quicksum(transport_plan[node1, node2] for node1 in node_list) == prob_mass[node2] for node2 in node_list)

        # Wasserstein distance constraint
        model.addConstr(gp.quicksum(transport_plan[node1, node2] * self.geodata.get_dist(node1, node2) 
                                    for node1 in node_list for node2 in node_list) <= self.epsilon, "wasserstein_distance")
        
        model.setParam('OutputFlag', 0)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Extract the optimal solution
            prob_mass_solution = {node: prob_mass[node].X for node in node_list}
            transport_plan_solution = {(node1, node2): transport_plan[node1, node2].X for node1 in node_list for node2 in node_list}
            return prob_mass_solution, transport_plan_solution
        else:
            raise Exception("Model did not find an optimal solution.")


    def evaluate_fixed_route(self, prob_dict):
        """
        Evalaute the costs incurred by riders and the provider using fixed-route mode.
        For riders, the costs include
            1) Walk time, expected walk distance from home to the nearest stop divided by the average walk speed 1.4 m/s
            2) Wait time, expected wait time at the stop until the shuttle arrives
            3) Transit time, expected transit time from the stop to the destination on the shuttle
        For the provider, the cost is
            1) Travel cost, the long-run average cost of the shuttle operting on a district

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
            raise NotImplementedError("TSP mode is not implemented yet.")
        
