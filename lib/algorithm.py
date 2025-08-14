import numpy as np
import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.data import GeoData
import logging
import concurrent.futures
from functools import partial
import multiprocessing as mp


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
        
        # Precompute and cache distance matrix for all blocks (expensive operation)
        print("Precomputing distance matrix for CQCP subproblems...")
        self._precompute_distance_matrix()
        print("Distance matrix cached successfully.")

    def _precompute_distance_matrix(self):
        """
        Precompute the distance matrix between all blocks to avoid recalculating 
        in every CQCP subproblem call.
        """
        N = len(self.short_geoid_list)
        self.distance_matrix = np.zeros((N, N))
        
        # Cache the mapping from block_id to index for faster lookups
        self.block_to_index = {block_id: i for i, block_id in enumerate(self.short_geoid_list)}
        
        # Precompute all pairwise distances
        for i, bi in enumerate(self.short_geoid_list):
            for j, bj in enumerate(self.short_geoid_list):
                self.distance_matrix[i, j] = self.geodata.get_dist(bi, bj)
                
        # Also precompute area vector for efficiency
        try:
            self.area_vector = np.array([float(self.geodata.get_area(bi)) for bi in self.short_geoid_list])
        except Exception:
            self.area_vector = np.ones(N)
            
        print(f"   - Cached {N}x{N} distance matrix")
        print(f"   - Cached area vector for {N} blocks")


    def _Hess_model(self, block_centers):

        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"

        assignment_indices = [(node, center) for node in self.short_geoid_list for center in block_centers]

        Hess_model = gp.Model("Hess_model")
        z = Hess_model.addVars(assignment_indices, vtype=GRB.BINARY, name="z")
        Hess_model.addConstrs((gp.quicksum(z[node, center] for center in block_centers) == 1 for node in self.short_geoid_list), name="unique_assignment")
        Hess_model.addConstrs((z[center, center] == 1 for center in block_centers), name="center_selection")
        # Use cached distance matrix for better performance
        objective_terms = []
        for node in self.short_geoid_list:
            node_idx = self.block_to_index[node]
            for center in block_centers:
                center_idx = self.block_to_index[center]
                objective_terms.append(z[node, center] * self.distance_matrix[node_idx, center_idx])
        Hess_model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
        Hess_model.setParam('OutputFlag', 0)
        Hess_model.optimize()

        block_assignment = np.array([[z[node, center].X for center in block_centers] for node in self.short_geoid_list])
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

            new_center = blocks_in_district[np.argmax([center_x[i].X for i in blocks_in_district])]
            new_centers.append(new_center)

        return np.array(new_centers)


    def _CQCP(self, z, block_centers):
        '''
        Solve the CQCP for lower bound of the problem
        '''
        obj_dict = {} # Dict to store the objective values for each district center: value
        node_list = self.short_geoid_list
        n = len(node_list)

        for district_idx in range(self.num_districts):

            model = gp.Model("inner_problem_lower_bound")
            # Add variables
            x = model.addVars(node_list, lb=0.0, name="x")
            y = model.addVars(node_list, node_list, lb=0.0, name='y')

            # Fix: Ensure proper type conversion for math.sqrt by converting to float
            area_values = []
            for i in range(n):
                area_val = self.geodata.get_area(node_list[i])
                area_values.append(float(area_val))
            model.setObjective(gp.quicksum(x[node_list[i]] * math.sqrt(area_values[i]) * z[i, district_idx] for i in range(n)), GRB.MAXIMIZE)

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

            # Fix: Use addConstr instead of addQConstr for linear constraints
            model.addConstr(gp.quicksum(x[node] for node in node_list) == 1, name="total_mass")
            model.addConstrs((gp.quicksum(y[node1, node2] for node2 in node_list) == self.prob_dict[node1] for node1 in node_list), name='y_sum')
            for node2 in node_list:
                model.addConstr((gp.quicksum(y[node1, node2] for node1 in node_list) == x[node2]), name='y_sumj')

            model.addConstr(gp.quicksum(self.geodata.get_dist(node1, node2) * y[node1, node2] for node1 in node_list for node2 in node_list) <= self.epsilon, name='wasserstein')
            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            district_area = np.sum([self.geodata.get_area(node_list[i]) * z[i, district_idx] for i in range(n)])
            obj_dict[block_centers[district_idx]] = {'bhh': model.objVal, 'district_area': district_area}

        return obj_dict
    
    def _expand_assignment_matrix(self, block_assignment, block_centers):
        """
        Expand a (N_blocks, num_districts) assignment matrix to (N_blocks, N_blocks),
        where columns corresponding to block_centers are filled, others are zero.
        """
        all_blocks = self.short_geoid_list
        N = len(all_blocks)
        num_districts = len(block_centers)
        expanded = np.zeros((N, N))
        center_idx_map = {center: all_blocks.index(center) for center in block_centers}
        for j in range(N):
            for d in range(num_districts):
                expanded[j, center_idx_map[block_centers[d]]] = block_assignment[j, d]
        return expanded

    def random_search(self, max_iters=1000, prob_dict=None, Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120):
        """
        Randomized search for the best partition using the real-problem objective.
        Returns best_block_centers, best_assignment, best_obj_val, best_district_info
        """
        best_obj_val = float('inf')
        best_block_centers = None
        best_assignment = None
        best_district_info = None
        block_ids = self.short_geoid_list
        N = len(block_ids)
        for _ in range(max_iters):
            block_centers = np.random.choice(block_ids, self.num_districts, replace=False)
            block_assignment = self._Hess_model(block_centers)
            new_centers = self._recenter(block_centers, block_assignment)
            new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
            cnt = 0
            while sorted(block_centers) != sorted(new_centers):
                block_centers = new_centers
                block_assignment = self._Hess_model(block_centers)
                new_centers = self._recenter(block_centers, block_assignment)
                new_centers = [self.short_geoid_list[int(center)] for center in new_centers]
                cnt += 1
            if block_assignment.shape[1] != N:
                expanded_assignment = self._expand_assignment_matrix(block_assignment, block_centers)
            else:
                expanded_assignment = block_assignment
            obj_val, district_info = self.evaluate_real_objective(expanded_assignment, prob_dict, Lambda, wr, wv, beta)
            if obj_val < best_obj_val:
                best_obj_val = obj_val
                best_block_centers = block_centers
                best_assignment = expanded_assignment
                best_district_info = district_info
        print(f"Best block centers: {best_block_centers}")
        print(f"Best objective value: {best_obj_val}")
        return best_block_centers, best_assignment, best_obj_val, best_district_info

    def local_search(self, block_centers, best_obj_val, prob_dict=None, Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120):
        """
        Local search for the best partition using the real-problem objective.
        Returns best_block_centers, best_assignment, best_obj_val, best_district_info
        """
        block_centers = list(block_centers)
        assert len(block_centers) == self.num_districts, f"Got {len(block_centers)} block centers but expected {self.num_districts} districts"
        assignment = self._Hess_model(block_centers)
        N = len(self.short_geoid_list)
        if assignment.shape[1] != N:
            expanded_assignment = self._expand_assignment_matrix(assignment, block_centers)
        else:
            expanded_assignment = assignment
        obj_val, district_info = self.evaluate_real_objective(expanded_assignment, prob_dict, Lambda, wr, wv, beta)
        print(f"Original objective value: {obj_val}")
        best_assignment = expanded_assignment
        best_obj_val = obj_val
        best_district_info = district_info
        for center in block_centers:
            for neighbor in self.geodata.G.neighbors(center):
                if neighbor not in block_centers:
                    new_centers = list(block_centers.copy())
                    new_centers.remove(center)
                    new_centers.append(neighbor)
                    new_assignment = self._Hess_model(new_centers)
                    if new_assignment.shape[1] != N:
                        expanded_new_assignment = self._expand_assignment_matrix(new_assignment, new_centers)
                    else:
                        expanded_new_assignment = new_assignment
                    new_obj_val, new_district_info = self.evaluate_real_objective(expanded_new_assignment, prob_dict, Lambda, wr, wv, beta)
                    if new_obj_val < best_obj_val:
                        print(f"new obj val: {new_obj_val}")
                        block_centers = new_centers
                        best_obj_val = new_obj_val
                        best_assignment = expanded_new_assignment
                        best_district_info = new_district_info
                        return self.local_search(block_centers, best_obj_val, prob_dict, Lambda, wr, wv, beta)
        return block_centers, best_assignment, best_obj_val, best_district_info

    def _CQCP_benders(self, assigned_blocks, root, prob_dict, epsilon, grid_points=20, beta=None, Lambda=None):
        """
        Solve the inner CQCP for a district (for Benders decomposition):
        - assigned_blocks: list of block IDs assigned to this district
        - root: the root block ID (must be in assigned_blocks)
        - prob_dict: probability mass for each block in the entire service region
        - epsilon: Wasserstein radius
        Returns:
            - worst-case cost (float)
            - optimal mass x_j^* for each block (dict)
            - subgradient components (g_alpha, g_K, etc.)
            - T_star (optimal dispatch interval)
            - alpha_i (sum of x^*_j for assigned blocks)
        
        Note: The SDP is formulated over the ENTIRE service region, but only assigned blocks
        contribute to the objective function through the assignment indicators z_ji.
        """
        # Use ALL blocks in the service region for the CQCP formulation
        all_blocks = self.short_geoid_list
        N = len(all_blocks)
        
        # Create assignment indicator: z_ji = 1 if block j is assigned to this district
        z_assignment = np.zeros(N)
        for j, block_id in enumerate(all_blocks):
            if block_id in assigned_blocks:
                z_assignment[j] = 1.0
        
        # Use precomputed distance matrix (much faster!)
        cost_matrix = self.distance_matrix
        
        # Empirical distribution over ALL blocks
        p = np.array([prob_dict.get(bi, 0.0) for bi in all_blocks])
        
        # Use precomputed area vector
        area_vec = self.area_vector
        
        # SDP: maximize sum of x_j * sqrt(area_j) * z_ji (only assigned blocks contribute)
        model = gp.Model("worst_case_qcp")
        x = model.addVars(N, lb=0.0, name="x")
        y = model.addVars(N, N, lb=0.0, name='y')
        
        # Objective: only assigned blocks contribute (multiply by z_assignment)
        model.setObjective(gp.quicksum(x[j] * np.sqrt(area_vec[j]) * z_assignment[j] for j in range(N)), gp.GRB.MAXIMIZE)
        
        # Constraints over ALL blocks
        # Quadratic constraint: sum of x_j^2 <= 1
        model.addQConstr(gp.quicksum(x[j] * x[j] for j in range(N)) <= 1, name="quad_constraint")
        
        # Mass conservation: sum_l y_jl = m_j for all j
        model.addConstrs((gp.quicksum(y[j, l] for l in range(N)) == p[j] for j in range(N)), name='mass_conservation')
        
        # Coupling constraint: sum_j y_jl >= x_l^2 for all l
        for l in range(N):
            model.addQConstr((gp.quicksum(y[j, l] for j in range(N)) >= x[l] * x[l]), name=f'coupling_{l}')
        
        # Wasserstein constraint: sum_{j,l} d_jl * y_jl <= epsilon
        model.addConstr(gp.quicksum(cost_matrix[j, l] * y[j, l] for j in range(N) for l in range(N)) <= epsilon, name='wasserstein')
        
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        if model.status != gp.GRB.OPTIMAL:
            logging.warning("SDP infeasible or not optimal!")
            return float('inf'), {bi: 0.0 for bi in all_blocks}, {bi: 0.0 for bi in all_blocks}, None, 0.0
        
        # Extract solution
        x_star = np.array([x[j].X for j in range(N)])
        
        # alpha_i is the optimal objective value of the CQCP with BHH coefficient: 
        # α_i = β * √Λ * Σ x_j * sqrt(area_j) * z_ji
        # Get BHH coefficient β and overall arrival rate Λ
        if beta is None:
            beta = getattr(self.geodata, 'beta', 0.7120)  # Default BHH coefficient
        if Lambda is None:
            Lambda = getattr(self.geodata, 'Lambda', 30.0)  # Default overall arrival rate
        
        # Raw alpha without BHH factors
        raw_alpha = np.sum(x_star * np.sqrt(area_vec) * z_assignment)
        # Apply BHH coefficient: α_i = β * √Λ * raw_alpha
        alpha_i = beta * np.sqrt(Lambda) * raw_alpha
        # Note: model.objVal gives raw_alpha, not the BHH-adjusted alpha_i
        
        # Retrieve K_i, F_i from GeoData instance
        try:
            K_i = float(self.geodata.get_K(root))
        except AttributeError:
            K_i = 0.0
            # logging.warning(f"GeoData missing get_K for block {root}, using K_i={K_i}")
        try:
            F_i = float(self.geodata.get_F(root))
        except AttributeError:
            F_i = 0.0
            # logging.warning(f"GeoData missing get_F for block {root}, using F_i={F_i}")
        
        wr = getattr(self.geodata, 'wr', 1.0)
        wv = getattr(self.geodata, 'wv', 10.0)
        
        # 1-D minimization in T (dispatch interval)
        Tmin = 1e-3
        Tmax = 10.0
        best_T = Tmin
        best_obj = float('inf')
        for T in np.linspace(Tmin, Tmax, grid_points):
            ci = np.sqrt(T)
            g_bar = (K_i+F_i)*ci**-2 + alpha_i*ci**-1 + wr/(2*wv)*K_i + wr/wv*alpha_i*ci + wr*ci**2
            if g_bar < best_obj:
                best_obj = g_bar
                best_T = T
        
        T_star = best_T
        ci_star = np.sqrt(T_star)
        g_alpha = ci_star**-1 + wr/wv*ci_star
        g_K = ci_star**-2 + wr/(2*wv)
        g_F = ci_star**-2
        
        # Subgradient: g_j for ALL blocks (not just assigned ones)
        # Since α_i = β * √Λ * Σ x_j * sqrt(area_j) * z_ji,
        # the subgradient w.r.t. z_ji is: ∂α_i/∂z_ji = β * √Λ * x_j^* * sqrt(area_j)
        subgrad = {}
        for j, bi in enumerate(all_blocks):
            area_j = self.geodata.get_area(bi)
            sqrt_area_j = np.sqrt(area_j)
            # Include BHH factors in subgradient
            dalpha_dzji = beta * np.sqrt(Lambda) * x_star[j] * sqrt_area_j
            if bi == root:
                subgrad[bi] = g_alpha * dalpha_dzji + ((K_i + F_i) * ci_star**-2 + wr/(2*wv) * K_i + wr * ci_star**2)
            else:
                subgrad[bi] = g_alpha * dalpha_dzji
        
        logging.info(f"CQCP district root {root}: cost={best_obj:.4f}, alpha={alpha_i:.4f}, T*={T_star:.4f}")
        
        # Return x_star for ALL blocks (not just assigned ones)
        # Also return subgradients w.r.t. K_i and F_i for master problem
        subgrad_K_i = g_K  # ∂g/∂K_i = ci_star**-2 + wr/(2*wv)
        subgrad_F_i = g_F  # ∂g/∂F_i = ci_star**-2
        
        return best_obj, dict(zip(all_blocks, x_star)), subgrad, T_star, alpha_i, subgrad_K_i, subgrad_F_i

    def _CQCP_benders_updated(self, assigned_blocks, root, prob_dict, epsilon, K_i, F_i, grid_points=20, beta=None, Lambda=None, single_threaded=False):
        """
        Updated CQCP subproblem that accepts partition-dependent costs K_i and F_i as parameters.
        
        Parameters:
        - assigned_blocks: list of block IDs assigned to this district
        - root: the root block ID (must be in assigned_blocks)
        - prob_dict: probability mass for each block in the entire service region
        - epsilon: Wasserstein radius
        - K_i: linehaul cost for this district (from master problem)
        - F_i: ODD cost for this district (from master problem)
        - grid_points: number of points for 1-D optimization over T
        
        Returns:
            - worst-case cost (float)
            - optimal mass x_j^* for each block (dict)
            - subgradient components w.r.t. master variables
            - T_star (optimal dispatch interval)
            - alpha_i (sum of x^*_j for assigned blocks)
        """
        # Use ALL blocks in the service region for the SDP formulation
        all_blocks = self.short_geoid_list
        N = len(all_blocks)
        
        # Create assignment indicator: z_ji = 1 if block j is assigned to this district
        z_assignment = np.zeros(N)
        for j, block_id in enumerate(all_blocks):
            if block_id in assigned_blocks:
                z_assignment[j] = 1.0
        
        # Use precomputed distance matrix (much faster!)
        cost_matrix = self.distance_matrix
        
        # Empirical distribution over ALL blocks
        p = np.array([prob_dict.get(bi, 0.0) for bi in all_blocks])
        
        # Use precomputed area vector
        area_vec = self.area_vector
        
        # SDP: maximize sum of x_j * sqrt(area_j) * z_ji (only assigned blocks contribute)
        model = gp.Model("worst_case_qcp")
        x = model.addVars(N, lb=0.0, name="x")
        y = model.addVars(N, N, lb=0.0, name='y')
        
        # Objective: only assigned blocks contribute (multiply by z_assignment)
        model.setObjective(gp.quicksum(x[j] * np.sqrt(area_vec[j]) * z_assignment[j] for j in range(N)), gp.GRB.MAXIMIZE)
        
        # Constraints over ALL blocks
        # Quadratic constraint: sum of x_j^2 <= 1
        model.addQConstr(gp.quicksum(x[j] * x[j] for j in range(N)) <= 1, name="quad_constraint")
        
        # Mass conservation: sum_l y_jl = m_j for all j
        model.addConstrs((gp.quicksum(y[j, l] for l in range(N)) == p[j] for j in range(N)), name='mass_conservation')
        
        # Coupling constraint: sum_j y_jl >= x_l^2 for all l
        for l in range(N):
            model.addQConstr((gp.quicksum(y[j, l] for j in range(N)) >= x[l] * x[l]), name=f'coupling_{l}')
        
        # Wasserstein constraint: sum_{j,l} d_jl * y_jl <= epsilon
        model.addConstr(gp.quicksum(cost_matrix[j, l] * y[j, l] for j in range(N) for l in range(N)) <= epsilon, name='wasserstein')
        
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        if model.status != gp.GRB.OPTIMAL:
            logging.warning("CQCP infeasible or not optimal!")
            return float('inf'), {bi: 0.0 for bi in all_blocks}, {bi: 0.0 for bi in all_blocks}, None, 0.0
        
        # Extract solution
        x_star = np.array([x[j].X for j in range(N)])
        
        # alpha_i is the optimal objective value of the CQCP with BHH coefficient: 
        # α_i = β * √Λ * Σ x_j * sqrt(area_j) * z_ji
        # Get BHH coefficient β and overall arrival rate Λ
        if beta is None:
            beta = getattr(self.geodata, 'beta', 0.7120)  # Default BHH coefficient
        if Lambda is None:
            Lambda = getattr(self.geodata, 'Lambda', 30.0)  # Default overall arrival rate
        
        # Raw alpha without BHH factors
        raw_alpha = np.sum(x_star * np.sqrt(area_vec) * z_assignment)
        # Apply BHH coefficient: α_i = β * √Λ * raw_alpha
        alpha_i = beta * np.sqrt(Lambda) * raw_alpha
        
        # Use the partition-dependent costs K_i and F_i from master problem
        wr = getattr(self.geodata, 'wr', 1.0)
        wv = getattr(self.geodata, 'wv', 10.0)
        
        # 1-D minimization in C (dispatch subinterval)
        Cmin = 1e-3
        Cmax = 12.0
        best_C = Cmin
        best_obj = float('inf')
        for C in np.linspace(Cmin, Cmax, grid_points):
            ci = np.sqrt(C)
            # Updated objective using partition-dependent costs
            g_bar = ((K_i + F_i) * ci**-2 + wr/(2*wv) * K_i + wr * ci**2) + (ci**-1 + wr/wv * ci) * alpha_i
            if g_bar < best_obj:
                best_obj = g_bar
                best_C = C
        
        C_star = best_C
        ci_star = np.sqrt(C_star)
        
        # Subgradients w.r.t. master problem variables
        # ∂g/∂α_i
        g_alpha = ci_star**-1 + wr/wv * ci_star
        # ∂g/∂K_i  
        g_K = ci_star**-2 + wr/(2*wv)
        # ∂g/∂F_i
        g_F = ci_star**-2
        
        # Subgradient w.r.t. assignment variables z_ji
        # Since α_i = β * √Λ * Σ_j x_j^* * sqrt(area_j) * z_ji, we have ∂α_i/∂z_ji = β * √Λ * x_j^* * sqrt(area_j)
        subgrad = {}
        for j, bi in enumerate(all_blocks):
            area_j = self.geodata.get_area(bi)
            sqrt_area_j = np.sqrt(area_j)
            # Include BHH factors in subgradient
            dalpha_dzji = beta * np.sqrt(Lambda) * x_star[j] * sqrt_area_j
            # Subgradient w.r.t. z_ji includes contribution through α_i
            subgrad[bi] = g_alpha * dalpha_dzji
            if bi == root:
                subgrad[bi] += (K_i + F_i) * ci**-2 + wr/(2*wv) * K_i + wr * ci**2
        
        # Subgradients w.r.t. K_i and F_i for master problem optimization
        subgrad_K_i = g_K  # ∂g/∂K_i = ci_star**-2 + wr/(2*wv)
        subgrad_F_i = g_F  # ∂g/∂F_i = ci_star**-2
        
        logging.info(f"CQCP district root {root}: cost={best_obj:.4f}, alpha={alpha_i:.4f}, C*={C_star:.4f}, K_i={K_i:.4f}, F_i={F_i:.4f}")
        
        # Return x_star for ALL blocks (not just assigned ones)
        # Also return subgradients w.r.t. K_i and F_i for master problem
        return best_obj, dict(zip(all_blocks, x_star)), subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i

    def _solve_subproblem_parallel(self, args):
        """
        Wrapper function for parallel subproblem solving
        """
        assigned_block_ids, root_id, prob_dict, epsilon, K_i, F_i, J_function, omega_sol = args
        
        # Calculate actual F_i using the J_function and the district's ODD vector
        actual_F_i = J_function(omega_sol)
        
        # Call subproblem solver with thread-safe flag
        return self._CQCP_benders_updated(
            assigned_block_ids, root_id, prob_dict, epsilon, 
            K_i=K_i, F_i=actual_F_i, single_threaded=True
        )

    def benders_decomposition(self, max_iterations=50, tolerance=1e-3, verbose=True, 
                            Omega_dict=None, J_function=None, parallel=True):
        """
        Updated LBBD with depot location decisions and partition-dependent costs.
        Uses multi-cut generation for all active districts (no limit on number of cuts).
        
        Parameters:
        - max_iterations: maximum number of iterations
        - tolerance: convergence tolerance for gap between upper and lower bounds
        - verbose: whether to print debug information
        - Omega_dict: dict mapping block_id to Omega_j (2D ODD feature vector)
        - J_function: function that takes 2D omega_i vector and returns F_i (ODD cost)
        - parallel: whether to use parallel subproblem solving (default True)
        """
        block_ids = self.short_geoid_list  # short_GEOID strings
        N = len(block_ids)
        k = self.num_districts
        best_partition = None
        best_cost = float('inf')
        best_depot = None
        best_K = None
        best_F = None
        best_omega = None
        lower_bound = -float('inf')
        upper_bound = float('inf')
        cuts = []  # Each cut: (cut_constant, cut_coeffs, cut_id)
        cut_id_counter = 0
        cuts_added_to_model = 0  # Track how many cuts have been added to the model
        iteration = 0
        history = []
        arc_list = self.geodata.get_arc_list()
        
        # Use original epsilon throughout (no scaling)
        
        # Default parameters if not provided
        if Omega_dict is None:
            Omega_dict = {block_id: np.array([1.0, 1.0]) for block_id in block_ids}  # Default 2D ODD
        if J_function is None:
            J_function = lambda omega: 0.1 * np.sum(omega) if hasattr(omega, '__len__') else 0.1 * omega  # Default linear ODD cost
        
        # Determine ODD dimensionality
        first_omega = next(iter(Omega_dict.values()))
        odd_dim = len(first_omega) if hasattr(first_omega, '__len__') else 1
            
        if not hasattr(self.geodata, 'arc_list') or not arc_list:
            raise ValueError("GeoData arc_list is empty or not initialized. Check GeoData initialization.")
        if verbose:
            print("Starting Benders decomposition...")
            print(f"Using {len(arc_list)} directed arcs from GeoData")
            print(f"Block IDs: {len(block_ids)} blocks")
            missing_arcs = []
            for block_id in block_ids:
                in_arcs = self.geodata.get_in_arcs(block_id)
                out_arcs = self.geodata.get_out_arcs(block_id)
                if not in_arcs and not out_arcs:
                    missing_arcs.append(block_id)
            if missing_arcs:
                print(f"Warning: {len(missing_arcs)} blocks have no arcs: {missing_arcs[:5]}...")
            else:
                print("All blocks have arc information")
        # --- Create master problem ONCE ---
        master = gp.Model("benders_master")
        
        # Decision variables
        z = master.addVars(N, N, vtype=GRB.BINARY, name="assignment")  # z[j, i]: assign block j to root i
        delta = master.addVars(N, vtype=GRB.BINARY, name="depot")  # δ_j: depot location at block j
        o = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="objval")  # Maximum district cost
        
        # 2D ODD variables: ω_i is now a vector for each district
        omega = master.addVars(N, odd_dim, lb=0.0, vtype=GRB.CONTINUOUS, name="omega")  # ω_i[d]: ODD summary dimension d for district i
        
        K = master.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="K")  # K_i: linehaul cost for district i
        F = master.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name="F")  # F_i: ODD cost for district i
        f = master.addVars(block_ids, arc_list, lb=0.0, vtype=GRB.CONTINUOUS, name="flows")
        
        # Constraints from formulation (7)
        # (7c) Exactly one depot
        master.addConstr(gp.quicksum(delta[j] for j in range(N)) == 1, name='one_depot')
        
        # (7d) Each block assigned to exactly one district
        master.addConstrs((gp.quicksum(z[j, i] for i in range(N)) == 1 for j in range(N)), name='one_assignment')
        
        # (7e) Exactly k districts
        master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name='num_districts')
        
        # (7f) Can only assign to active roots
        master.addConstrs((z[j, i] <= z[i, i] for i in range(N) for j in range(N)), name='break_symmetry')
        
        # (7g) Linehaul cost K_i = 2d(κ_i, Σ_j κ_j δ_j) where κ_i is centroid of root i
        # For each potential root i, compute distance to depot
        for root_idx, root_id in enumerate(block_ids):
            # K_i = 2 * distance from root i to depot location
            # Use cached distance matrix for linehaul cost calculation
            dist_expr = gp.quicksum(2 * self.distance_matrix[root_idx, j] * delta[j] for j in range(N))
            master.addConstr(K[root_idx] == dist_expr, name=f'linehaul_cost_{root_id}')
        
        # (7h) ODD cost F_i = J(ω_i) - use linear J_function directly
        # Since J_function is linear and zero at zero, we can use it directly
        for i in range(N):
            if odd_dim >= 2:
                # For 2D case: F_i = J([ω_i[0], ω_i[1]]) - linear function
                # Test J_function at unit vectors to get coefficients
                try:
                    # Get linear coefficients by evaluating at unit vectors
                    j_at_zero = J_function(np.zeros(odd_dim))  # Should be 0
                    j_coeff_0 = J_function(np.array([1.0] + [0.0] * (odd_dim - 1))) - j_at_zero
                    j_coeff_1 = J_function(np.array([0.0, 1.0] + [0.0] * (odd_dim - 2))) - j_at_zero
                    
                    # Linear constraint: F_i = j_coeff_0 * ω_i[0] + j_coeff_1 * ω_i[1] + ...
                    if odd_dim == 2:
                        master.addConstr(F[i] == j_coeff_0 * omega[i, 0] + j_coeff_1 * omega[i, 1], 
                                       name=f'odd_cost_linear_{i}')
                    else:
                        # General case for higher dimensions
                        expr = j_at_zero
                        for d in range(odd_dim):
                            unit_vec = np.zeros(odd_dim)
                            unit_vec[d] = 1.0
                            j_coeff = J_function(unit_vec) - j_at_zero
                            expr += j_coeff * omega[i, d]
                        master.addConstr(F[i] == expr, name=f'odd_cost_linear_{i}')
                        
                except Exception:
                    # Fallback: assume simple linear form F_i = w1*ω_i[0] + w2*ω_i[1]
                    w1, w2 = 1.2, 0.8
                    master.addConstr(F[i] == w1 * omega[i, 0] + w2 * omega[i, 1], 
                                   name=f'odd_cost_linear_{i}')
            else:
                # For 1D case: F_i = J(ω_i[0]) - linear function
                try:
                    j_at_zero = J_function(np.array([0.0]))  # Should be 0
                    j_coeff = J_function(np.array([1.0])) - j_at_zero
                    master.addConstr(F[i] == j_coeff * omega[i, 0], name=f'odd_cost_linear_{i}')
                except Exception:
                    # Fallback: assume coefficient of 1.0
                    master.addConstr(F[i] == omega[i, 0], name=f'odd_cost_linear_{i}')
        
        # (7i) ODD constraint: ω_i[d] >= max{Ω_j[d] : j ∈ district i}
        # Efficient implementation: O(N²) instead of O(N³) constraints
        
        # Pre-compute all ODD values and find global maximum for big-M
        all_omega_values = []
        for j in range(N):
            if hasattr(Omega_dict[block_ids[j]], '__len__'):
                all_omega_values.extend(Omega_dict[block_ids[j]])
            else:
                all_omega_values.append(Omega_dict[block_ids[j]])
        M_omega = max(all_omega_values) if all_omega_values else 10.0
        
        for i in range(N):  # For each district
            for d in range(odd_dim):  # For each ODD dimension
                # ω_i[d] must be at least the maximum ODD value of any assigned block
                for j in range(N):  # For each block
                    if hasattr(Omega_dict[block_ids[j]], '__len__'):
                        omega_jd = Omega_dict[block_ids[j]][d]
                    else:
                        omega_jd = Omega_dict[block_ids[j]]
                    
                    # If block j is assigned to district i (z[j,i] = 1), then ω_i[d] >= Ω_j[d]
                    # If block j is not assigned (z[j,i] = 0), constraint is relaxed
                    master.addConstr(omega[i, d] >= omega_jd - M_omega * (1 - z[j, i]), 
                                   name=f'odd_max_{i}_{d}_{j}')
        
        # (7j) Ensure o >= K_i + F_i for all districts (basic lower bound)
        # This ensures o represents the maximum district cost and cannot be arbitrarily small
        # for i in range(N):
        #     master.addConstr(o >= K[i] + F[i], name=f'min_district_cost_{i}')
        
        # TODO: Add logical constraints later after testing basic single-cut functionality
        # Logical constraints: If district i is not selected (z_{ii} = 0), then K_i = 0 and F_i = 0
        # This prevents the master problem from setting arbitrary values for unused districts
        # M_K = 1000.0  # Big-M for K_i (should be larger than any reasonable linehaul cost)
        # M_F = 100.0   # Big-M for F_i (should be larger than any reasonable ODD cost)
        # 
        # for i in range(N):
        #     # K_i <= M_K * z_{ii} (if district i not selected, then K_i = 0)
        #     master.addConstr(K[i] <= M_K * z[i, i], name=f'logical_K_{i}')
        #     # F_i <= M_F * z_{ii} (if district i not selected, then F_i = 0)  
        #     master.addConstr(F[i] <= M_F * z[i, i], name=f'logical_F_{i}')
        #     # ω_i[d] <= M_ω * z_{ii} for all dimensions d
        #     for d in range(odd_dim):
        #         M_omega = 10.0  # Big-M for omega (should be larger than any reasonable ODD feature value)
        #         master.addConstr(omega[i, d] <= M_omega * z[i, i], name=f'logical_omega_{i}_{d}')
        
        # Flow-based contiguity constraints (7j, 7k, 7l, 7m)
        for i, root_id in enumerate(block_ids):
            # (7l) No return flow to root
            master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_in_arcs(root_id)) == 0, name=f"no_return_flow_{root_id}")
            for j, block_id in enumerate(block_ids):
                if block_id != root_id:
                    # (7j) Flow conservation: f^i(δ^-(j)) - f^i(δ^+(j)) = z_ji
                    in_flow = gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_in_arcs(block_id))
                    out_flow = gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_out_arcs(block_id))
                    master.addConstr(in_flow - out_flow == z[j, i], name=f"flow_assign_{block_id}_{root_id}")
                    # (7k) Flow capacity: f^i(δ^-(j)) <= (N-1) * z_ji
                    master.addConstr(in_flow <= (N - 1) * z[j, i], name=f"flow_restrict_{block_id}_{root_id}")
        
        master.setObjective(o, GRB.MINIMIZE)
        master.setParam('OutputFlag', 0)
        # --- End static model creation ---
        
        # Main LBBD iteration loop
        while iteration < max_iterations and (upper_bound - lower_bound > tolerance):
            if verbose:
                print(f"\nIteration {iteration+1}")
                print(f"  [DEBUG] While loop condition: iter={iteration} < {max_iterations} and gap={upper_bound - lower_bound:.4f} > {tolerance}")
                
                # --- Add all pending cuts from previous iteration ---
                if verbose:
                    print(f"  [DEBUG] Processing cuts: cuts_added_to_model={cuts_added_to_model}, total_cuts={len(cuts)}, new_cuts={len(cuts[cuts_added_to_model:])}")
                
                new_cuts_added = 0
                for cut_info in cuts[cuts_added_to_model:]:
                    cut_constant = cut_info['constant']
                    cut_coeffs_z = cut_info['z_coeffs']
                    K_coeff = cut_info['K_coeff']
                    F_coeff = cut_info['F_coeff']
                    district = cut_info['district']
                    cut_id = cut_info['id']
                
                    # Build cut expression: o >= constant + Σ(coeff_z * z_ji) + coeff_K * K_district + coeff_F * F_district
                    cut_expr = cut_constant
                
                    # Add z coefficients
                    if cut_coeffs_z:
                        cut_expr += gp.quicksum(cut_coeffs_z[j, i] * z[j, i] for (j, i) in cut_coeffs_z.keys())
                
                    # Add K coefficient (if non-zero)
                    if K_coeff != 0:
                        cut_expr += K_coeff * K[district]
                
                    # Add F coefficient (if non-zero)  
                    if F_coeff != 0:
                        cut_expr += F_coeff * F[district]
                
                    # Add cut to master problem: o >= cost_of_district_i
                    master.addConstr(o >= cut_expr, name=f"cut_{cut_id}")
                    new_cuts_added += 1
                
                    if verbose:
                        z_terms = " + ".join([f"{v:.4f}*z[{j},{i}]" for (j,i),v in cut_coeffs_z.items()])
                        K_term = f" + {K_coeff:.4f}*K[{district}]" if K_coeff != 0 else ""
                        F_term = f" + {F_coeff:.4f}*F[{district}]" if F_coeff != 0 else ""
                        print(f"  [DEBUG] Added cut_{cut_id}: o >= {cut_constant:.4f} + {z_terms}{K_term}{F_term}")
                
                # Update cuts_added_to_model AFTER adding cuts to master
                cuts_added_to_model += new_cuts_added
                if verbose and new_cuts_added > 0:
                    print(f"  [DEBUG] Updated cuts_added_to_model: {cuts_added_to_model-new_cuts_added} -> {cuts_added_to_model}")
                
                # Optimize master problem after adding all pending cuts
                if verbose:
                    print(f"  [DEBUG] Optimizing master problem with {cuts_added_to_model} cuts in model...")
                master.optimize()
                if master.status != GRB.OPTIMAL:
                    if master.status == GRB.INFEASIBLE:
                        logging.error("Master problem is infeasible")
                        master.computeIIS()
                        master.write(f"infeasible_master_iter_{iteration}.ilp")
                        raise RuntimeError(f"Master problem infeasible at iteration {iteration}")
                    else:
                        logging.error(f"Master problem failed with status {master.status}")
                        raise RuntimeError(f"Master problem failed with status {master.status} at iteration {iteration}")
                # Extract solution
                z_sol = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        z_sol[j, i] = z[j, i].X
                o_sol = o.X
            
                # Extract depot location
                depot_sol = [j for j in range(N) if round(delta[j].X) == 1]
                if len(depot_sol) != 1:
                    raise RuntimeError(f"Invalid depot solution: {len(depot_sol)} depots selected")
                depot_block = block_ids[depot_sol[0]]
            
                # Extract partition-dependent costs
                K_sol = [K[i].X for i in range(N)]
                F_sol = [F[i].X for i in range(N)]
            
                # Extract 2D ODD vectors
                omega_sol = []
                for i in range(N):
                    if odd_dim >= 2:
                        omega_sol.append(np.array([omega[i, d].X for d in range(odd_dim)]))
                    else:
                        omega_sol.append(omega[i, 0].X)
            
                roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
                district_costs = []
                subgrads = []
            
                # Generate cuts for existing districts (selected roots)
                # Use parallel solving only when it's beneficial (>=3 districts, sufficient problem size)
                use_parallel = (parallel and len(roots) >= 3 and 
                              sum(len([j for j in range(N) if round(z_sol[j, root]) == 1]) for root in roots) >= 12)
                
                if use_parallel:
                    # Parallel subproblem solving
                    subproblem_args = []
                    for root in roots:
                        assigned_blocks = [j for j in range(N) if round(z_sol[j, root]) == 1]
                        assigned_block_ids = [block_ids[j] for j in assigned_blocks]
                        district_omega = omega_sol[root]
                        args = (assigned_block_ids, block_ids[root], self.prob_dict, self.epsilon, 
                               K_sol[root], F_sol[root], J_function, district_omega)
                        subproblem_args.append((args, root, assigned_blocks))
                
                    # Solve subproblems in parallel with timing
                    import time
                    parallel_start = time.time()
                    
                    # Use fewer workers to reduce contention (max 2 for most cases)
                    max_workers = min(2, len(roots))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for args, root, assigned_blocks in subproblem_args:
                            future = executor.submit(self._solve_subproblem_parallel, args)
                            futures.append((future, root, assigned_blocks))
                        
                        # Collect results
                        for future, root, assigned_blocks in futures:
                            cost, x_star, subgrad, T_star, alpha_i, subgrad_K_i, subgrad_F_i = future.result()
                            district_costs.append((cost, root, assigned_blocks, subgrad, x_star, T_star, alpha_i, subgrad_K_i, subgrad_F_i))
                            subgrads.append(subgrad)
                    
                    parallel_time = time.time() - parallel_start
                    if verbose:
                        print(f"  Solved {len(roots)} subproblems in parallel ({parallel_time:.3f}s, {max_workers} workers)")
                else:
                    # Serial subproblem solving (fallback or single district)
                    import time
                    serial_start = time.time()
                    
                    for root in roots:
                        assigned_blocks = [j for j in range(N) if round(z_sol[j, root]) == 1]
                        # Pass the partition-dependent costs to subproblem
                        # Compute actual F_i using the J_function and the district's ODD vector
                        district_omega = omega_sol[root]
                        actual_F_i = J_function(district_omega)
                    
                        cost, x_star, subgrad, T_star, alpha_i, subgrad_K_i, subgrad_F_i = self._CQCP_benders_updated(
                            [block_ids[j] for j in assigned_blocks], block_ids[root], 
                            self.prob_dict, self.epsilon, K_i=K_sol[root], F_i=actual_F_i)
                        district_costs.append((cost, root, assigned_blocks, subgrad, x_star, T_star, alpha_i, subgrad_K_i, subgrad_F_i))
                        subgrads.append(subgrad)
                    
                    serial_time = time.time() - serial_start
                    if verbose:
                        print(f"  Solved {len(roots)} subproblems serially ({serial_time:.3f}s)")
            
                # Multi-cut Benders: Generate cuts for ALL active districts
                cuts_added = 0
                worst_cost = 0.0  # Track the maximum district cost for upper bound
            
                # Generate a Benders cut for each active district
                for cost, root, assigned_blocks, subgrad, x_star_dist, T_star_dist, alpha_i_dist, subgrad_K_i_dist, subgrad_F_i_dist in district_costs:
                    # Update worst cost for upper bound calculation
                    worst_cost = max(worst_cost, cost)
                
                    # Get the values that were actually passed to this district's subproblem
                    K_i_prev = K_sol[root]  # K_i value passed to subproblem
                    district_omega_prev = omega_sol[root]
                    F_i_prev = J_function(district_omega_prev)  # F_i value passed to subproblem
                
                    # 1. Coefficients and RHS values for partition variables z_ji
                    cut_coeffs_z = {}
                    cut_rhs_val_z = 0.0
                    for j in range(N):
                        g_j = subgrad.get(block_ids[j], 0.0)
                        if g_j != 0.0:
                            cut_coeffs_z[j, root] = g_j
                            # Use the z value that was passed to the subproblem (current master solution)
                            cut_rhs_val_z += g_j * z_sol[j, root]
                
                    # 2. Coefficients and RHS values for linehaul cost K_i and ODD cost F_i
                    K_coeff = subgrad_K_i_dist  # ∂g/∂K_i
                    F_coeff = subgrad_F_i_dist  # ∂g/∂F_i
                
                    # 3. Calculate constant term correctly for Benders cut
                    # The cut is: o >= c* + g_z*(z - z*) + g_K*(K - K*) + g_F*(F - F*)
                    # Which becomes: o >= (c* - g_z*z* - g_K*K* - g_F*F*) + g_z*z + g_K*K + g_F*F
                    # So the constant term is: c* - g_z*z* - g_K*K* - g_F*F*
                    cut_rhs_val_K = K_coeff * K_i_prev if K_coeff != 0 else 0.0
                    cut_rhs_val_F = F_coeff * F_i_prev if F_coeff != 0 else 0.0
                    cut_constant = cost - cut_rhs_val_z - cut_rhs_val_K - cut_rhs_val_F
                
                    # 4. Store complete cut information (z coefficients, K coefficient, F coefficient)
                    cut_info = {
                        'constant': cut_constant,
                        'z_coeffs': cut_coeffs_z,
                        'K_coeff': K_coeff,
                        'F_coeff': F_coeff,
                        'district': root,
                        'id': cut_id_counter
                    }
                    cuts.append(cut_info)
                
                    if verbose:
                        z_terms = " + ".join([f"{v:.4f}*z[{j},{i}]" for (j, i), v in cut_coeffs_z.items()])
                        K_term = f" + {K_coeff:.4f}*K[{root}]" if K_coeff != 0 else ""
                        F_term = f" + {F_coeff:.4f}*F[{root}]" if F_coeff != 0 else ""
                        print(f"  [DEBUG] Will add cut_{cut_id_counter} next iter: o >= {cut_constant:.4f} + {z_terms}{K_term}{F_term}")
                
                    cut_id_counter += 1
                    cuts_added += 1
            
            if verbose:
                print(f"  Total cuts generated this iter: {cuts_added} (multi-cut approach for all {len(district_costs)} active districts)")
                print(f"  [DEBUG] Reached cut generation end - cuts will be added next iteration")
            
            # Lower bound is the optimal objective value of the master problem
            lower_bound = master.objVal
            upper_bound = min(upper_bound, worst_cost)
            if verbose:
                print(f"  [DEBUG] Bounds calculated: LB={lower_bound:.4f}, UB={upper_bound:.4f}, Gap={upper_bound-lower_bound:.4f}")
            
            # Debug: Check why lower bound might be 0
            if verbose and lower_bound == 0.0:
                print(f"  [DEBUG] Master status: {master.status}, objVal: {master.objVal:.6f}")
                print(f"  [DEBUG] o.X = {o.X:.6f}, Number of cuts: {len(cuts)}")
                if len(cuts) > 0:
                    print(f"  [DEBUG] First cut constant: {cuts[0]['constant']:.6f}")
            
            if verbose:
                print(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}, Gap: {upper_bound - lower_bound:.4f}")
            if worst_cost < best_cost:
                best_cost = worst_cost
                best_partition = z_sol.copy()
                best_depot = depot_block
                best_K = K_sol.copy()
                best_F = F_sol.copy()
                best_omega = omega_sol.copy()
            history.append({
                'iteration': iteration+1,
                'z_sol': z_sol.copy(),
                'depot_block': depot_block,
                'K_sol': K_sol.copy(),
                'F_sol': F_sol.copy(),
                'omega_sol': omega_sol.copy(),
                'o_sol': o_sol,
                'worst_cost': worst_cost,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'gap': upper_bound - lower_bound,
                'roots': roots,
                'district_costs': district_costs
            })
            iteration += 1
            
        if verbose:
            print("\nBenders decomposition finished.")
            if parallel:
                print("✅ Parallel subproblem solving enabled")
            print(f"Best cost: {best_cost:.4f}")
            print(f"Best depot: {best_depot}")
        
        # Return comprehensive results including depot and partition-dependent costs
        return {
            'best_partition': best_partition,
            'best_cost': best_cost,
            'best_depot': best_depot,
            'best_K': best_K,
            'best_F': best_F,
            'best_omega': best_omega,
            'history': history,
            'converged': (upper_bound - lower_bound <= tolerance),
            'iterations': iteration,
            'final_gap': upper_bound - lower_bound
        }

    def evaluate_real_objective(self, assignment, prob_dict, Lambda, wr, wv, beta=0.7120):
        """
        Evaluate the real-problem objective for a given assignment (z matrix):
        - assignment: np.ndarray (N_blocks x N_blocks), z[j, i]=1 if block j assigned to root i
        - prob_dict: empirical distribution
        - Lambda, wr, wv: parameters for the objective
        - beta: risk parameter (default 0.7120)
        Returns: max district cost, list of tuples (district_obj, root, K_i, T_star)
        """
        block_ids = self.short_geoid_list
        N = len(block_ids)
        epsilon = self.epsilon
        district_info = []
        for i, root in enumerate(block_ids):
            # Get assigned blocks for this root
            assigned = [j for j in range(N) if round(assignment[j, i]) == 1]
            if not assigned or round(assignment[i, i]) != 1:
                continue  # skip if not a root or no blocks assigned
            assigned_blocks = [block_ids[j] for j in assigned]
            # CQCP inner maximization
            cost, x_star, _, T_star, alpha_i, subgrad_K_i, subgrad_F_i = self._CQCP_benders(assigned_blocks, root, prob_dict, epsilon)
            # Get K_i for the root
            K_i = float(self.geodata.get_K(root))
            # F_i = 0
            F_i = 0.0
            # C_i = sqrt(T_star)
            C_i = np.sqrt(T_star) if T_star is not None else 1.0
            # Compute the full objective as in (8a)
            term1 = (K_i + F_i) / C_i
            term2 = beta * np.sqrt(Lambda / C_i) * alpha_i
            term3 = wr * (1/wv * (K_i/2 + beta * np.sqrt(Lambda * C_i) * alpha_i) + C_i)
            district_obj = term1 + term2 + term3
            district_info.append((district_obj, root, K_i, T_star))
        if not district_info:
            return float('inf'), []
        max_cost = max([info[0] for info in district_info])
        return max_cost, district_info