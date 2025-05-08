import numpy as np
import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Partition:
    def __init__(self, geodata, num_districts, epsilon=0.0001):
        """
        Initialize the Partition object.

        Parameters:
        - data: GeoData object
        - num_partitions: int, number of partitions to create
        """
        self.geodata = geodata
        self.gdf = geodata.gdf
        self.short_geoid_list = geodata.short_geoid_list
        self.shortest_distance_dict = geodata.shortest_distance_dict
        self.area_dict = geodata.gdf['area'].to_dict()
        self.num_districts = num_districts
        self.epsilon = epsilon

    def Hess_model(self, block_centers):

        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"

        assignment_indices = [(bg, center) for bg in self.short_geoid_list for center in block_centers]

        Hess_model = gp.Model("Hess_model")
        z = Hess_model.addVars(assignment_indices, vtype=GRB.BINARY, name="z")
        Hess_model.addConstrs((gp.quicksum(z[bg, center] for center in block_centers) == 1 for bg in self.short_geoid_list), name="unique_assignment")
        Hess_model.addConstrs((z[center, center] == 1 for center in block_centers), name="center_selection")
        Hess_model.setObjective(gp.quicksum(z[bg, center] * self.shortest_distance_dict[(bg, center)] for bg in self.short_geoid_list for center in block_centers), GRB.MINIMIZE)
        Hess_model.setParam('OutputFlag', 0)
        Hess_model.optimize()

        block_assignment = np.array([[z[bg, center].x for center in block_centers] for bg in self.short_geoid_list])
        return block_assignment


    def recenter(self, block_centers, block_assignment):
        '''
        Recenter the block centers based on the block assignment such that the overall
        travel distance from blocks in that district to the center is minimized
        '''
        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"
        new_centers = []

        for old_center_idx in range(self.num_districts):
            
            blocks_in_district = np.where(block_assignment[:, old_center_idx] == 1)[0]
            recenter_model = gp.Model("recenter_model")
            center_x = recenter_model.addVars(blocks_in_district, vtype=GRB.BINARY, name="z")
            recenter_model.addConstr(gp.quicksum(center_x[i] for i in blocks_in_district) == 1)
            recenter_model.setObjective(gp.quicksum(center_x[i] * self.shortest_distance_dict[(self.short_geoid_list[i], self.short_geoid_list[j])] for i in blocks_in_district for j in blocks_in_district), GRB.MINIMIZE)
            recenter_model.setParam('OutputFlag', 0)
            recenter_model.optimize()

            new_center = blocks_in_district[np.argmax([center_x[i].x for i in blocks_in_district])]
            new_centers.append(new_center)

        return np.array(new_centers)


    def SDP(self, z, block_centers, r):
        '''
        Solve the SDP for lower bound of the problem
        '''

        obj_dict = {} # Dict to store the objective values for each district center: value
        n = len(self.short_geoid_list)

        for district_idx in range(self.num_districts):

            model = gp.Model("inner_problem_lower_bound")
            # Add variables
            x = model.addVars(self.short_geoid_list, lb=0.0, name="x")
            y = model.addVars(self.short_geoid_list, self.short_geoid_list, lb=0.0, name='y')

            # Set objective: maximize sum of x_i
            model.setObjective(gp.quicksum(x[self.short_geoid_list[i]] * math.sqrt(self.area_dict[self.short_geoid_list[i]]) * z[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

            # Add quadratic constraint: sum of squares of x_i <= 1
            model.addQConstr(gp.quicksum(x[bg] * x[bg] for bg in self.short_geoid_list) <= 1, name="quad_constraint")
            model.addConstrs((gp.quicksum(y[bg1, bg2] for bg2 in self.short_geoid_list) == r[bg1] for bg1 in self.short_geoid_list), name='y_sum')
            for bg2 in self.short_geoid_list:
                model.addQConstr((gp.quicksum(y[bg1, bg2] for bg1 in self.short_geoid_list) >= x[bg2] * x[bg2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.shortest_distance_dict[(bg1, bg2)] * y[bg1, bg2] for bg1 in self.short_geoid_list for bg2 in self.short_geoid_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            district_prob = np.sum([r[self.short_geoid_list[i]] * z[i, district_idx] for i in range(n)])
            obj_dict[block_centers[district_idx]] = {'bhh': model.objVal, 'district_mass': district_prob}

        return obj_dict


    def LP(self, z, block_centers, r):
        '''
        Solve the LP for upper bound of the problem
        '''

        obj_dict = {} # Dict to store the objective values for each district center: value
        n = len(self.short_geoid_list)

        for district_idx in range(self.num_districts):

            model = gp.Model("inner_problem_upper_bound")
            # Add variables
            x = model.addVars(self.short_geoid_list, lb=0.0, name="x")
            y = model.addVars(self.short_geoid_list, self.short_geoid_list, lb=0.0, name='y')

            # Set objective: maximize sum of x_i
            model.setObjective(gp.quicksum(x[self.short_geoid_list[i]] * z[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

            # Add quadratic constraint: sum of squares of x_i <= 1
            model.addQConstr(x.sum() == 1, name="total_mass")
            model.addConstrs((gp.quicksum(y[bg1, bg2] for bg2 in self.short_geoid_list) == r[bg1] for bg1 in self.short_geoid_list), name='y_sum')
            for bg2 in self.short_geoid_list:
                model.addQConstr((gp.quicksum(y[bg1, bg2] for bg1 in self.short_geoid_list) == x[bg2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.shortest_distance_dict[(bg1, bg2)] * y[bg1, bg2] for bg1 in self.short_geoid_list for bg2 in self.short_geoid_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            district_area = np.sum([self.area_dict[self.short_geoid_list[i]] * z[i, district_idx] for i in range(n)])
            obj_dict[block_centers[district_idx]] = {'bhh': model.objVal, 'district_area': district_area}

        return obj_dict
    
    def random_search(self, max_iters=1000):
        # Randomized Heuristic per Validi, Buchanan, and Lykhovyd, 2022

        num_blocks = len(self.gdf)

        worst_district_list = []
        district_costs_list = []
        block_centers_list = []
        best_obj_val_lb = float('inf')
        best_obj_val = float('inf')
        best_block_centers = None

        for _ in range(max_iters):
            block_centers = np.random.choice(self.gdf.index, self.num_districts, replace=False)
            block_assignment = self.Hess_model(block_centers)
            new_centers = self.recenter(block_centers, block_assignment)
            new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
            cnt = 0
            print(f"ITERATION {cnt}: Centers updated from {block_centers} to {new_centers}")
            while sorted(block_centers) != sorted(new_centers):
                block_centers = new_centers
                block_assignment = self.Hess_model(block_centers)
                new_centers = self.recenter(block_centers, block_assignment)
                new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
                cnt += 1
                print(f"ITERATION {cnt}: Centers updated from {block_centers} to {new_centers}")

            # Calculate the lower bound of the objective value for all districts and the worst district
            obj_dict_lb = self.SDP(block_assignment, block_centers, probability_dict)
            district_lb_costs = [obj_dict_lb[center]['bhh']**2 * obj_dict_lb[center]['district_mass'] for center in block_centers]
            worst_district_lb = max(obj_dict_lb.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_mass'])
            worst_district_cost_lb = worst_district_lb[1]['bhh']**2 * worst_district_lb[1]['district_mass']

            # Calculate the upper bound of the objective value for all districts and the worst district
            obj_dict_ub = self.LP(block_assignment, block_centers, probability_dict)
            district_ub_costs = [obj_dict_ub[center]['bhh']**2 * obj_dict_ub[center]['district_area'] for center in block_centers]
            worst_district_ub = max(obj_dict_ub.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_area'])
            worst_district_cost_ub = worst_district_ub[1]['bhh']**2 * worst_district_ub[1]['district_area']

            print(f"The gap between the lower and upper bound is {worst_district_cost_ub - worst_district_cost_lb}")
            
            worst_district_list.append({'lb': (worst_district_lb, worst_district_cost_lb), 'ub': (worst_district_ub, worst_district_cost_ub)})
            district_costs_list.append({'lb': district_lb_costs, 'ub': district_ub_costs})
            block_centers_list.append(block_centers)
            if worst_district_cost_ub < best_obj_val:
                best_obj_val = worst_district_cost_ub
                best_block_centers = block_centers

    def localsearch(self):
        pass