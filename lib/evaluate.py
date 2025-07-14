import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import polyline
import simpy
from shapely.geometry import LineString, Point

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


    # ==========================================================
    # Fixed-route related methods
    # ==========================================================

    def get_fixed_route_worst_distribution(self, prob_dict):
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
        

    def find_worst_walk_node(self, dest_name='Walmart (Garden Center)'):
        """
        Identify the node with the maximal walk distance to its nearest stop,
        and compute its expected wait (headway/2) and transit time to the destination.

        Returns:
            dict with keys:
              'node', 'walk_distance', 'route', 'wait_time', 'transit_time'
        """
        nearest = self.routedata.find_nearest_stops()
        # Worst node by walk distance
        worst = max(nearest, key=lambda n: nearest[n]['distance'])
        info = nearest[worst]
        walk_dist = info['distance']
        route_name = info['route']
        boarding_stop = info['stop']
        # Retrieve route stops
        route = next(rt for rt in self.routedata.routes_info
                     if rt.get('Description', f"Route {rt.get('RouteID')}") == route_name)
        stops = route.get('Stops', [])
        secs = [s.get('SecondsToNextStop', 0) for s in stops]
        names = [s.get('Description', '') for s in stops]
        headway = sum(secs)
        wait_time = headway / 2.0
        # Find indices
        board_idx = next(i for i, s in enumerate(stops) if s == boarding_stop)
        dest_idx = next((i for i, nm in enumerate(names) if dest_name in nm), None)
        if dest_idx is None:
            transit_time = None
        else:
            if dest_idx >= board_idx:
                transit_time = sum(secs[board_idx:dest_idx])
            else:
                transit_time = sum(secs[board_idx:]) + sum(secs[:dest_idx])
        return {
            'node': worst,
            'walk_distance': walk_dist,
            'route': route_name,
            'wait_time': wait_time,
            'transit_time': transit_time
        }


    def simulate_wait_and_transit_fixed_route(self, prob_dict, n_sims=10000, dest_name="Walmart (Garden Center)"):
        """
        Already replaced by simulate_wait_and_transit_fixed_route_simpy.

        Simulate rider wait and transit times under fixed-route service mode using uniformly random bus locations.

        Assumes each rider travels from their home node, boards at the nearest stop,
        and all riders unboard at the specified destination stop.

        Parameters
        ----------
        prob_dict : dict
            A dictionary that maps the node index to the true or worst probability mass.
        n_sims : int
            Number of riders to simulate.
        dest_name : str
            Name of the common destination stop.

        Returns
        -------
        wait_times : np.ndarray
            Simulated wait times (seconds) for each rider.
        transit_times : np.ndarray
            Simulated in-vehicle transit times (seconds) for each rider.
        """
        # 1) Get nearest stop info per node
        nearest = self.routedata.find_nearest_stops()  # node -> {'route', 'stop', 'distance'}

        # 2) Build sampling distribution
        nodes = list(nearest.keys())
        probs = np.array([prob_dict.get(n, 0.0) for n in nodes])
        probs = probs / probs.sum()

        # 3) Sample riders
        sampled = np.random.choice(nodes, size=n_sims, p=probs)
        wait_times = np.zeros(n_sims)
        transit_times = np.zeros(n_sims)

        # 4) Simulate each rider
        for i, node in enumerate(sampled):
            info = nearest[node]
            route_name = info['route']
            boarding_stop = info['stop']

            # Find route and its stop sequence
            for route in self.routedata.routes_info:
                if route.get("Description", f"Route {route.get('RouteID')}") == route_name:
                    stops = route.get("Stops", [])
                    # extract segment times and names
                    secs = [s.get("SecondsToNextStop", 0) for s in stops]
                    names = [s.get("Description", "") for s in stops]

                    # indices
                    board_idx = next(idx for idx, s in enumerate(stops) if s == boarding_stop)
                    dest_idx = names.index(dest_name) if dest_name in names else None

                    # 4a) Wait: uniform on headway
                    headway = sum(secs)
                    wait_times[i] = np.random.rand() * headway

                    # 4b) Transit: sum travel times from boarding to destination
                    if dest_idx is None:
                        transit_times[i] = np.nan
                    else:
                        if dest_idx >= board_idx:
                            transit_times[i] = sum(secs[board_idx:dest_idx])
                        else:
                            transit_times[i] = sum(secs[board_idx:]) + sum(secs[:dest_idx])
                    break

        return wait_times, transit_times


    def simulate_wait_and_transit_fixed_route_simpy(self, prob_dict, sim_time=3600, dest_name='Walmart (Garden Center)'):
        """
        Simulate buses and Poisson rider arrivals using SimPy.

        Returns wait and transit summaries per route:
            wait_summary   : {route: (mass_assigned, avg_wait_seconds)}
            transit_summary: {route: (mass_assigned, avg_transit_seconds)}
        """
        # 0) Create simulation environment
        env = simpy.Environment()

        # 1) Find nearest stop for each node
        nearest = self.routedata.find_nearest_stops()

        # 2) Initialize stores and mass per route
        routes = self.routedata.routes_info
        stop_map = {}
        mass_per_route = {}
        for route in routes:
            rname = route.get('Description', f"Route {route.get('RouteID')}")
            mass_per_route[rname] = 0.0
            for i, s in enumerate(route.get('Stops', [])):
                stop_map[(rname, i)] = simpy.Store(env)

        # 3) Compute arrival rates and assign masses
        arrivals = {}
        for node, info in nearest.items():
            r = info['route']
            sdict = info['stop']
            seq = next(rt.get('Stops', []) for rt in routes if rt.get('Description') == r)
            idx = next(i for i, x in enumerate(seq) if x == sdict)
            mass = prob_dict.get(node, 0)
            mass_per_route[r] += mass
            arrivals[(r, idx)] = arrivals.get((r, idx), 0) + mass
        # Normalize rates so sum equals total_mass per hour
        total_mass = sum(prob_dict.values())
        for key in arrivals:
            arrivals[key] = arrivals[key]  # keep mass as relative rate

        # 4) Define processes
        wait_records = {r: [] for r in mass_per_route}
        transit_records = {r: [] for r in mass_per_route}

        def rider_gen(route, idx, rate, store):
            while True:
                # Poisson interarrival
                yield env.timeout(np.random.exponential(1.0 / rate))
                store.put({'time': env.now, 'route': route, 'stop_idx': idx})

        def bus_proc(route):
            stops = next(rt.get('Stops') for rt in routes if rt.get('Description') == route)
            secs = [s.get('SecondsToNextStop', 0) for s in stops]
            names = [s.get('Description', '') for s in stops]
            dest_idx = next((i for i, nm in enumerate(names) if nm == dest_name), None)
            i = 0
            while env.now < sim_time:
                store = stop_map[(route, i)]
                # discharge riders
                while store.items:
                    rider = yield store.get()
                    wait = env.now - rider['time']
                    wait_records[route].append(wait)
                    # compute transit
                    if dest_idx is not None:
                        if dest_idx >= i:
                            ttime = sum(secs[i:dest_idx])
                        else:
                            ttime = sum(secs[i:]) + sum(secs[:dest_idx])
                        transit_records[route].append(ttime)
                # move to next
                yield env.timeout(secs[i])
                i = (i + 1) % len(stops)

        # 5) Start processes
        for (r, idx), rate in arrivals.items():
            if rate > 0:
                env.process(rider_gen(r, idx, rate, stop_map[(r, idx)]))
        for r in mass_per_route:
            env.process(bus_proc(r))

        # 6) Run simulation
        env.run(until=sim_time)

        # 7) Summarize
        wait_summary = {r: (mass_per_route[r], np.mean(wait_records[r]) if wait_records[r] else 0)
                        for r in mass_per_route}
        transit_summary = {r: (mass_per_route[r], np.mean(transit_records[r]) if transit_records[r] else 0)
                           for r in mass_per_route}
        return wait_summary, transit_summary



    def evaluate_fixed_route(self, prob_dict, simulate=True):
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


        if simulate:
            # 4) Simulate wait and transit times
            wait_times, transit_times = self.simulate_wait_and_transit_fixed_route_simpy(prob_dict, sim_time=43200)
            expected_wait_time = np.sum([wait_times[rt][0] * wait_times[rt][1] for rt in wait_times])
            expected_transit_time = np.sum([transit_times[rt][0] * transit_times[rt][1] for rt in transit_times])

            return {
                'route_lengths': route_lengths,
                'travel_times': travel_times,
                'walk_time_detail': walk_time_detail,
                'expected_walk_time': expected_walk_time,
                'wait_time_detail': wait_times,
                'expected_wait_time': expected_wait_time,
                'transit_time_detail': transit_times,
                'expected_transit_time': expected_transit_time
            }

        
        else:
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

    # ==========================================================
    # Doorstep pickup (TSP) related methods
    # ==========================================================


    def get_tsp_worst_distributions(self, prob_dict, center_list, node_assignment):
        """
        Get the worst-case distributions for TSP mode on each district.
        The costs we need to consider include the costs on the riders: wait time and transit time; and the costs on the provider: travel cost.
        The worst-case distribution is the one that maximizes the overall, within the Wasserstein ball centered at the estimated empirical distribution.

        :param prob_dict: dict
            A dictionary that maps the node index to the estimated empirical true probability mass.
            The keys are the node indices, and the values are the probability masses.
        """
        node_list = self.short_geoid_list
        n = len(node_list)
        probability_dict_dict = {}

        for center in center_list:
            district_idx = center_list.index(center)

            model = gp.Model("inner_problem_lower_bound")
            # Add variables
            x = model.addVars(node_list, lb=0.0, name="x")
            y = model.addVars(node_list, node_list, lb=0.0, name='y')

            # Set objective: maximize sum of x_i
            model.setObjective(gp.quicksum(x[node_list[i]] * math.sqrt(self.geodata.get_area(node_list[i])) * node_assignment[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

            # Add quadratic constraint: sum of squares of x_i <= 1
            model.addQConstr(gp.quicksum(x[node] * x[node] for node in node_list) <= 1, name="quad_constraint")
            model.addConstrs((gp.quicksum(y[node1, node2] for node2 in node_list) == prob_dict[node1] for node1 in node_list), name='y_sum')
            for node2 in node_list:
                model.addQConstr((gp.quicksum(y[node1, node2] for node1 in node_list) >= x[node2] * x[node2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.geodata.get_dist(node1, node2) * y[node1, node2] for node1 in node_list for node2 in node_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()


            if model.status == GRB.OPTIMAL:
                # Extract the optimal solution
                prob_mass_solution = {node: (x[node].X)**2 for node in node_list}
                # transport_plan_solution = {(node1, node2): y[node1, node2].X for node1 in node_list for node2 in node_list}
                probability_dict_dict[center] = prob_mass_solution

            else:
                raise Exception("Model did not find an optimal solution.")

        return probability_dict_dict

    def evaluate_tsp_mode_on_single_district(self, prob_dict, node_assignment, center_list, center, unit_wait_cost=1.0, overall_arrival_rate=1000, max_dipatch_interval=24):
        """
        Evaluate the costs incurred by riders and the provider using TSP mode on a single district
        with its corresponding worst-case distribution dict.
        For riders, the costs include
            1) Wait time, expected wait time at home until the shuttle arrives
            2) Transit time, expected transit time from home to the destination on the shuttle
        For the provider, the cost is
            1) Travel cost, the long-run average cost of the shuttle operting on a district

        Calculate wait and transit times for TSP mode.
        Returns a dict with detailed and expected metrics.
        """
        node_list = self.short_geoid_list
        n = len(node_list)
        district_idx = center_list.index(center)

        # Integrate the probability mass on the district
        district_prob = np.sum([prob_dict[node_list[i]] * node_assignment[i, district_idx] for i in range(n)])
        district_arrival_rate = overall_arrival_rate * district_prob
        # Integrate the square root of the probability mass on the district, we use the vanilla implementation which can be optimized via
        # the optimal objective value obatined from get_tsp_worst_distributions
        district_prob_sqrt = np.sum([math.sqrt(prob_dict[node_list[i]]) * math.sqrt(self.geodata.get_area(node_list[i])) * node_assignment[i, district_idx] for i in range(n)])
        # Get the optimal dispatch interval using 1-D minimization as in Partition._SDP_benders
        Tmin = 1e-3
        Tmax = max_dipatch_interval  # Use argument as the upper bound for search
        grid_points = 20
        best_T = Tmin
        best_obj = float('inf')
        # For legacy compatibility, set K_i and F_i to 0 (or use actual values if available)
        try:
            K_i = float(self.geodata.get_K(center))
        except Exception:
            K_i = 0.0
        try:
            F_i = float(self.geodata.get_F(center))
        except Exception:
            F_i = 0.0
        wr = getattr(self.geodata, 'wr', 1.0)
        wv = getattr(self.geodata, 'wv', 10.0)
        alpha_i = district_prob_sqrt  # for compatibility with Partition convention
        for T in np.linspace(Tmin, Tmax, grid_points):
            ci = np.sqrt(T)
            g_bar = (K_i+F_i)*ci**-2 + alpha_i*ci**-1 + wr/(2*wv)*K_i + wr/wv*alpha_i*ci + wr*ci**2
            if g_bar < best_obj:
                best_obj = g_bar
                best_T = T
        interval = best_T

        mean_wait_time_per_interval_per_rider = interval / 2
        mean_transit_distance_per_interval = 2287 / (math.sqrt(214) * 210) * math.sqrt(overall_arrival_rate * interval) * district_prob_sqrt

        # amt_wait_time = mean_wait_time_per_interval_per_rider / interval
        amt_transit_distance = mean_transit_distance_per_interval / interval

        results_dict = {
            'district_center': center,
            'district_prob': district_prob,
            'dispatch_interval': interval,
            'district_prob_sqrt': district_prob_sqrt,
            'mean_wait_time_per_interval_per_rider': mean_wait_time_per_interval_per_rider,
            'mean_transit_distance_per_interval': mean_transit_distance_per_interval,
            'amt_transit_distance': amt_transit_distance,
        }

        return results_dict

    def evaluate_tsp_mode(self, prob_dict, node_assignment, center_list, unit_wait_cost=1.0, overall_arrival_rate=1000, worst_case=True, max_dipatch_interval=24):
        """
        Evaluate the costs incurred by riders and the provider using TSP mode.
        For riders, the costs include
            1) Wait time, expected wait time at home until the shuttle arrives
            2) Transit time, expected transit time from home to the destination on the shuttle
        For the provider, the cost is
            1) Travel cost, the long-run average cost of the shuttle operting on a district

        Calculate wait and transit times for TSP mode.
        Returns a dict with detailed and expected metrics.
        """
        worst_probability_dict_dict = self.get_tsp_worst_distributions(prob_dict, center_list, node_assignment)
        results_dict = {}

        for center in center_list:
            if worst_case:
                # Get the worst-case distribution for TSP mode
                prob_dict = worst_probability_dict_dict[center]
            # Evaluate the costs incurred by riders and the provider using TSP mode
            results_dict[center] = self.evaluate_tsp_mode_on_single_district(prob_dict, node_assignment, center_list, center, unit_wait_cost, overall_arrival_rate, max_dipatch_interval)

        return results_dict



    def evaluate(self, node_assignment, center_list, prob_dict, mode="fixed", unit_wait_cost=None, overall_arrival_rate=None):
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
            assert unit_wait_cost is not None, "unit_wait_cost must not be None"
            assert overall_arrival_rate is not None, "overall_arrival_rate must not be None"
            results_dict = self.evaluate_tsp_mode(prob_dict, node_assignment, center_list, unit_wait_cost, overall_arrival_rate)
            return results_dict
        
