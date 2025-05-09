import numpy as np
import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.data import GeoData


class Partition:
    def __init__(self, geodata: GeoData, num_districts, prob_dict, epsilon=0.0001):
        """
        Initialize the Partition object.

        Parameters:
        - data: GeoData object
        - num_partitions: int, number of partitions to create
        """
        self.geodata = geodata
        self.gdf = geodata.gdf
        self.short_geoid_list = geodata.short_geoid_list
        self.num_districts = num_districts
        self.prob_dict = prob_dict
        self.epsilon = epsilon

    def _Hess_model(self, block_centers):

        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"

        assignment_indices = [(node, center) for node in self.short_geoid_list for center in block_centers]

        Hess_model = gp.Model("Hess_model")
        z = Hess_model.addVars(assignment_indices, vtype=GRB.BINARY, name="z")
        Hess_model.addConstrs((gp.quicksum(z[node, center] for center in block_centers) == 1 for node in self.short_geoid_list), name="unique_assignment")
        Hess_model.addConstrs((z[center, center] == 1 for center in block_centers), name="center_selection")
        Hess_model.setObjective(gp.quicksum(z[node, center] * self.geodata.get_dist(node, center) for node in self.short_geoid_list for center in block_centers), GRB.MINIMIZE)
        Hess_model.setParam('OutputFlag', 0)
        Hess_model.optimize()

        block_assignment = np.array([[z[node, center].x for center in block_centers] for node in self.short_geoid_list])
        return block_assignment


    def _recenter(self, block_centers, block_assignment):
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
            recenter_model.setObjective(gp.quicksum(center_x[i] * self.geodata.get_dist(self.short_geoid_list[i], self.short_geoid_list[j])
                                                    for i in blocks_in_district for j in blocks_in_district), GRB.MINIMIZE)
            recenter_model.setParam('OutputFlag', 0)
            recenter_model.optimize()

            new_center = blocks_in_district[np.argmax([center_x[i].x for i in blocks_in_district])]
            new_centers.append(new_center)

        return np.array(new_centers)


    def _SDP(self, z, block_centers):
        '''
        Solve the SDP for lower bound of the problem
        '''
        obj_dict = {} # Dict to store the objective values for each district center: value
        node_list = self.short_geoid_list
        n = len(node_list)

        for district_idx in range(self.num_districts):

            model = gp.Model("inner_problem_lower_bound")
            # Add variables
            x = model.addVars(node_list, lb=0.0, name="x")
            y = model.addVars(node_list, node_list, lb=0.0, name='y')

            # Set objective: maximize sum of x_i
            model.setObjective(gp.quicksum(x[node_list[i]] * math.sqrt(self.geodata.get_area(node_list[i])) * z[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

            # Add quadratic constraint: sum of squares of x_i <= 1
            model.addQConstr(gp.quicksum(x[node] * x[node] for node in node_list) <= 1, name="quad_constraint")
            model.addConstrs((gp.quicksum(y[node1, node2] for node2 in node_list) == self.prob_dict[node1] for node1 in node_list), name='y_sum')
            for node2 in node_list:
                model.addQConstr((gp.quicksum(y[node1, node2] for node1 in node_list) >= x[node2] * x[node2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.geodata.get_dist(node1, node2) * y[node1, node2] for node1 in node_list for node2 in node_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            district_prob = np.sum([self.prob_dict[node_list[i]] * z[i, district_idx] for i in range(n)])
            obj_dict[block_centers[district_idx]] = {'bhh': model.objVal, 'district_mass': district_prob}

        return obj_dict


    def _LP(self, z, block_centers):
        '''
        Solve the LP for upper bound of the problem
        '''

        obj_dict = {} # Dict to store the objective values for each district center: value
        node_list = self.short_geoid_list
        n = len(node_list)

        for district_idx in range(self.num_districts):

            model = gp.Model("inner_problem_upper_bound")
            # Add variables
            x = model.addVars(node_list, lb=0.0, name="x")
            y = model.addVars(node_list, node_list, lb=0.0, name='y')

            # Set objective: maximize sum of x_i
            model.setObjective(gp.quicksum(x[node_list[i]] * z[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

            # Add quadratic constraint: sum of squares of x_i <= 1
            model.addQConstr(x.sum() == 1, name="total_mass")
            model.addConstrs((gp.quicksum(y[node1, node2] for node2 in node_list) == self.prob_dict[node1] for node1 in node_list), name='y_sum')
            for node2 in node_list:
                model.addQConstr((gp.quicksum(y[node1, node2] for node1 in node_list) == x[node2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.geodata.get_dist(node1, node2) * y[node1, node2] for node1 in node_list for node2 in node_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            district_area = np.sum([self.geodata.get_area(node_list[i]) * z[i, district_idx] for i in range(n)])
            obj_dict[block_centers[district_idx]] = {'bhh': model.objVal, 'district_area': district_area}

        return obj_dict
    
    def random_search(self, max_iters=1000, criterion='lb'):
        worst_district_list = []
        district_costs_list = []
        block_centers_list = []
        best_obj_val_lb = float('inf')
        best_obj_val_ub = float('inf')
        best_block_centers = None

        for _ in range(max_iters):
            block_centers = np.random.choice(self.gdf.index, self.num_districts, replace=False)
            block_assignment = self._Hess_model(block_centers)
            new_centers = self._recenter(block_centers, block_assignment)
            new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
            cnt = 0
            print(f"ITERATION {cnt}: Centers updated from {block_centers} to {new_centers}")
            while sorted(block_centers) != sorted(new_centers):
                block_centers = new_centers
                block_assignment = self._Hess_model(block_centers)
                new_centers = self._recenter(block_centers, block_assignment)
                new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
                cnt += 1
                print(f"ITERATION {cnt}: Centers updated from {block_centers} to {new_centers}")

            # Calculate the lower bound of the objective value for all districts and the worst district
            obj_dict_lb = self._SDP(block_assignment, block_centers)
            district_lb_costs = [obj_dict_lb[center]['bhh']**2 * obj_dict_lb[center]['district_mass'] for center in block_centers]
            worst_district_lb = max(obj_dict_lb.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_mass'])
            worst_district_cost_lb = worst_district_lb[1]['bhh']**2 * worst_district_lb[1]['district_mass']

            # Calculate the upper bound of the objective value for all districts and the worst district
            obj_dict_ub = self._LP(block_assignment, block_centers)
            district_ub_costs = [obj_dict_ub[center]['bhh']**2 * obj_dict_ub[center]['district_area'] for center in block_centers]
            worst_district_ub = max(obj_dict_ub.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_area'])
            worst_district_cost_ub = worst_district_ub[1]['bhh']**2 * worst_district_ub[1]['district_area']

            print(f"The gap between the lower and upper bound is {worst_district_cost_ub - worst_district_cost_lb}")
            
            worst_district_list.append({'lb': (worst_district_lb, worst_district_cost_lb), 'ub': (worst_district_ub, worst_district_cost_ub)})
            district_costs_list.append({'lb': district_lb_costs, 'ub': district_ub_costs})
            block_centers_list.append(block_centers)

            if criterion == 'lb':
                if worst_district_cost_lb < best_obj_val_lb:
                    best_obj_val_lb = worst_district_cost_lb
                    best_block_centers = block_centers
            elif criterion == 'ub':
                if worst_district_cost_ub < best_obj_val_ub:
                    best_obj_val_ub = worst_district_cost_ub
                    best_block_centers = block_centers
        
        if criterion == 'lb':
            best_obj_val = best_obj_val_lb
        elif criterion == 'ub':
            best_obj_val = best_obj_val_ub

        print(f"Best block centers: {best_block_centers}")
        print(f"Best objective value: {best_obj_val}")
        best_assignment = self._Hess_model(best_block_centers)

        return best_block_centers, best_assignment, best_obj_val, worst_district_list, district_costs_list, block_centers_list
        

    def local_search(self, block_centers, best_obj_val, criterion='lb'):

        assert criterion in ['lb', 'ub'], "Criterion must be either 'lb' or 'ub'"
        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"

        assignment = self._Hess_model(block_centers)
        obj_dict = self._SDP(assignment, block_centers) if criterion == 'lb' else self._LP(assignment, block_centers)
        if criterion == 'lb':
            obj_val_dict = {node: obj_dict[node]['bhh']**2 * obj_dict[node]['district_mass'] for node in obj_dict.keys()}
            best_obj_val = max(obj_val_dict.values())
        elif criterion == 'ub':
            obj_val_dict = {node: obj_dict[node]['bhh']**2 * obj_dict[node]['district_area'] for node in obj_dict.keys()}
            best_obj_val = max(obj_val_dict.values())
        print(f"Original objective value: {best_obj_val}")

        for center in block_centers:
            for neighbor in self.geodata.G.neighbors(center):
                if neighbor not in block_centers:
                    new_centers = block_centers.copy()
                    new_centers.remove(center)
                    new_centers.append(neighbor)
                    new_assignment = self._Hess_model(new_centers)
                    if criterion == 'lb':
                        new_obj_dict = self._SDP(new_assignment, new_centers)
                        new_obj_val_dict = {node: new_obj_dict[node]['bhh']**2 * new_obj_dict[node]['district_mass'] for node in new_obj_dict.keys()}
                        new_obj_val = max(new_obj_val_dict.values())
                    elif criterion == 'ub':
                        new_obj_dict = self._LP(new_assignment, new_centers)
                        new_obj_val_dict = {node: new_obj_dict[node]['bhh']**2 * new_obj_dict[node]['district_area'] for node in new_obj_dict.keys()}
                        new_obj_val = max(new_obj_val_dict.values())
                    
                    if new_obj_val < best_obj_val:
                        print(f"new obj val: {new_obj_val}")
                        block_centers = new_centers
                        best_obj_val = new_obj_val
                        return self.local_search(block_centers, best_obj_val)
                    
        best_assignment = self._Hess_model(block_centers)
        return block_centers, best_assignment, best_obj_val