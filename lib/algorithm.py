import numpy as np
import gurobipy as gp
import math
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.data import GeoData
import logging


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
            # district_lb_costs = [obj_dict_lb[center]['bhh']**2 * obj_dict_lb[center]['district_mass'] for center in block_centers]
            # worst_district_lb = max(obj_dict_lb.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_mass'])
            # worst_district_cost_lb = worst_district_lb[1]['bhh']**2 * worst_district_lb[1]['district_mass']
            district_lb_costs = [obj_dict_lb[center]['bhh'] for center in block_centers]
            worst_district_lb = max(obj_dict_lb.items(), key=lambda x: x[1]['bhh'])
            worst_district_cost_lb = worst_district_lb[1]['bhh']

            # Calculate the upper bound of the objective value for all districts and the worst district
            obj_dict_ub = self._LP(block_assignment, block_centers)
            district_ub_costs = [obj_dict_ub[center]['bhh']**2 * obj_dict_ub[center]['district_area'] for center in block_centers]
            worst_district_ub = max(obj_dict_ub.items(), key=lambda x: x[1]['bhh']**2 * x[1]['district_area'])
            worst_district_cost_ub = worst_district_ub[1]['bhh']**2 * worst_district_ub[1]['district_area']

            # print(f"The gap between the lower and upper bound is {worst_district_cost_ub - worst_district_cost_lb}")
            
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
            obj_val_dict = {node: obj_dict[node]['bhh'] for node in obj_dict.keys()}
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
                        new_obj_val_dict = {node: new_obj_dict[node]['bhh'] for node in new_obj_dict.keys()}
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

    def _SDP_benders(self, assigned_blocks, root, prob_dict, epsilon, grid_points=20):
        """
        Solve the inner SDP for a district (for Benders decomposition):
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
        # Use ALL blocks in the service region for the SDP formulation
        all_blocks = self.short_geoid_list
        N = len(all_blocks)
        
        # Create assignment indicator: z_ji = 1 if block j is assigned to this district
        z_assignment = np.zeros(N)
        for j, block_id in enumerate(all_blocks):
            if block_id in assigned_blocks:
                z_assignment[j] = 1.0
        
        # Cost matrix over ALL blocks (N x N)
        cost_matrix = np.zeros((N, N))
        for i, bi in enumerate(all_blocks):
            for j, bj in enumerate(all_blocks):
                cost_matrix[i, j] = self.geodata.get_dist(bi, bj)
        
        # Empirical distribution over ALL blocks
        p = np.array([prob_dict.get(bi, 0.0) for bi in all_blocks])
        
        # Area vector over ALL blocks
        try:
            area_vec = np.array([float(self.geodata.get_area(bi)) for bi in all_blocks])
        except Exception:
            area_vec = np.ones(N)
        
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
        
        # alpha_i is the optimal objective value of the QCP: max Σ x_j * sqrt(area_j) * z_ji
        # This equals the sum of x_j^* * sqrt(area_j) for assigned blocks only
        alpha_i = np.sum(x_star * np.sqrt(area_vec) * z_assignment)
        # Equivalently: alpha_i = model.objVal
        
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
        
        # Subgradient: g_j for ALL blocks (not just assigned ones)
        # Since the SDP objective is max Σ x_j * sqrt(area_j) * z_ji,
        # the subgradient w.r.t. z_ji should include the sqrt(area_j) factor
        subgrad = {}
        for j, bi in enumerate(all_blocks):
            area_j = self.geodata.get_area(bi)
            sqrt_area_j = np.sqrt(area_j)
            if bi == root:
                subgrad[bi] = g_alpha * x_star[j] * sqrt_area_j + g_K * (K_i + F_i)
            else:
                subgrad[bi] = g_alpha * x_star[j] * sqrt_area_j
        
        logging.info(f"SDP district root {root}: cost={best_obj:.4f}, alpha={alpha_i:.4f}, T*={T_star:.4f}")
        
        # Return x_star for ALL blocks (not just assigned ones)
        return best_obj, dict(zip(all_blocks, x_star)), subgrad, T_star, alpha_i

    def benders_decomposition(self, max_iterations=50, tolerance=1e-3, max_cuts=100, verbose=True):
        block_ids = self.short_geoid_list  # short_GEOID strings
        N = len(block_ids)
        k = self.num_districts
        best_partition = None
        best_cost = float('inf')
        lower_bound = -float('inf')
        upper_bound = float('inf')
        cuts = []
        cut_id_counter = 0
        iteration = 0
        history = []
        
        # Get arc structures from GeoData
        arc_list = self.geodata.get_arc_list()
        
        # Validate arc structures
        if not hasattr(self.geodata, 'arc_list') or not arc_list:
            raise ValueError("GeoData arc_list is empty or not initialized. Check GeoData initialization.")
        
        if verbose:
            print("Starting Benders decomposition...")
            print(f"Using {len(arc_list)} directed arcs from GeoData")
            print(f"Block IDs: {len(block_ids)} blocks")
            
            # Validate that all blocks have arc information
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
        
        while iteration < max_iterations and (upper_bound - lower_bound > tolerance):
            if verbose:
                print(f"\nIteration {iteration+1}")
            master = gp.Model("benders_master")
            z = master.addVars(N, N, vtype=GRB.BINARY, name="assignment")  # z[j, i]: assign block j to root i (indices)
            o = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="objval")
            # Create flow variables for each root and each arc
            f = master.addVars(block_ids, arc_list, lb=0.0, vtype=GRB.CONTINUOUS, name="flows")


            if verbose and iteration == 0:
                print(f"Created {N*N} binary z variables and 1 objective variable")
            
            # Each block assigned to one district
            master.addConstrs((gp.quicksum(z[j, i] for i in range(N)) == 1 for j in range(N)), name='one_assignment')
            master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name='num_districts')
            master.addConstrs((z[j, i] <= z[i, i] for i in range(N) for j in range(N)), name='break_symmetry')
            
            if verbose and iteration == 0:
                print(f"Added basic assignment constraints: {N + 1 + N*N} constraints")
                print(f"Model has {master.NumVars} variables and {master.NumConstrs} constraints after basic constraints")
            # --- Add flow-based contiguity constraints (precise) ---
            for i, root_id in enumerate(block_ids):
                master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_in_arcs(root_id)) == 0, name=f"no_return_flow_{root_id}")
                for j, block_id in enumerate(block_ids):
                    if block_id != root_id:
                        master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_in_arcs(block_id)) - gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_out_arcs(block_id)) == z[j, i], name=f"flow_assign_{block_id}_{root_id}")
                        master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in self.geodata.get_in_arcs(block_id)) <= (N - 1) * z[j, i], name=f"flow_restrict_{block_id}_{root_id}")
            
            if verbose and iteration == 0:
                print(f"Created {len(f)} flow variables for {len(block_ids)} roots and {len(arc_list)} arcs")
                print(f"Model has {master.NumVars} variables and {master.NumConstrs} constraints after variable creation")
            
            if verbose and iteration == 0:
                print(f"Added flow constraints for contiguity")
                print(f"Model has {master.NumVars} variables and {master.NumConstrs} constraints after constraints")
            # --- Add Benders cuts ---
            cut_handles = []
            for cut_constant, cut_coeffs, cut_rhs, cut_id in cuts:
                # Recreate cut expression with current model variables
                cut_expr = cut_constant + gp.quicksum(cut_coeffs[j, i] * z[j, i] for (j, i) in cut_coeffs.keys()) - cut_rhs
                cut_handles.append(master.addConstr(o >= cut_expr, name=f"cut_{cut_id}"))
            master.setObjective(o, GRB.MINIMIZE)
            master.setParam('OutputFlag', 0)
            master.optimize()
            
            # Check optimization status
            if master.status != GRB.OPTIMAL:
                logging.error(f"Master problem failed to solve optimally. Status: {master.status}")
                if master.status == GRB.INFEASIBLE:
                    logging.error("Master problem is infeasible")
                    master.computeIIS()
                    master.write(f"infeasible_master_iter_{iteration}.ilp")
                    logging.error("IIS written to file for debugging")
                elif master.status == GRB.UNBOUNDED:
                    logging.error("Master problem is unbounded")
                elif master.status == GRB.INF_OR_UNBD:
                    logging.error("Master problem is infeasible or unbounded")
                else:
                    logging.error(f"Master problem optimization failed with status {master.status}")
                
                # Try to get some solution information if available
                try:
                    if master.SolCount > 0:
                        logging.info("Using best available solution")
                    else:
                        logging.error("No solution available")
                        raise RuntimeError(f"Master problem optimization failed with status {master.status}")
                except:
                    raise RuntimeError(f"Master problem optimization failed with status {master.status}")
            
            z_sol = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    z_sol[j, i] = z[j, i].X
            o_sol = o.X
            roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
            district_costs = []
            subgrads = []
            for root in roots:
                assigned_blocks = [j for j in range(N) if round(z_sol[j, root]) == 1]
                cost, x_star, subgrad, T_star, alpha_i = self._SDP_benders(
                    [block_ids[j] for j in assigned_blocks], block_ids[root], self.prob_dict, self.epsilon)
                district_costs.append((cost, root, assigned_blocks, subgrad, x_star, T_star, alpha_i))
                subgrads.append(subgrad)
            worst_idx = int(np.argmax([c[0] for c in district_costs]))
            worst_cost, worst_root, worst_blocks, worst_subgrad, x_star, T_star, alpha_i = district_costs[worst_idx]
            
            # Multi-cuts: Generate a cut for EVERY district, not just the worst one
            cuts_added = 0
            for cost, root, assigned_blocks, subgrad, x_star_dist, T_star_dist, alpha_i_dist in district_costs:
                # Store cut coefficients instead of expressions
                cut_constant = cost
                cut_coeffs = {}  # coefficients for z variables
                cut_rhs = 0.0
                for j in range(N):
                    g_j = subgrad.get(block_ids[j], 0.0)
                    if g_j != 0.0:
                        cut_coeffs[j, root] = g_j  # z[j,i] indexing
                        cut_rhs += g_j * z_sol[j, root]
                
                cuts.append((cut_constant, cut_coeffs, cut_rhs, cut_id_counter))
                cut_id_counter += 1
                cuts_added += 1
                if verbose:
                    print(f"  Added cut for district {block_ids[root]}: cost={cost:.4f}, T*={T_star_dist:.4f}, alpha={alpha_i_dist:.4f}")
            
            if verbose:
                print(f"  Total cuts added: {cuts_added} (multi-cuts approach)")
            if len(cuts) > max_cuts:
                violations = []
                for idx, (cut_constant, cut_coeffs, cut_rhs, cut_id) in enumerate(cuts):
                    # Calculate cut value at current solution
                    cut_val = cut_constant - sum(cut_coeffs[j, i] * z_sol[j, i] for (j, i) in cut_coeffs.keys()) + cut_rhs
                    slack = o_sol - cut_val
                    violations.append(abs(slack))
                idx_remove = int(np.argmin(violations))
                if verbose:
                    print(f"  Removing least violated cut: cut_{cuts[idx_remove][3]} (violation={violations[idx_remove]:.4e})")
                cuts.pop(idx_remove)
            lower_bound = o_sol
            upper_bound = min(upper_bound, worst_cost)
            if verbose:
                print(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}, Gap: {upper_bound - lower_bound:.4f}")
            if worst_cost < best_cost:
                best_cost = worst_cost
                best_partition = z_sol.copy()
            history.append({
                'iteration': iteration+1,
                'z_sol': z_sol.copy(),
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
            print(f"Best cost: {best_cost:.4f}")
        return best_partition, best_cost, history