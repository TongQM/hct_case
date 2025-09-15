"""
Run the real-case optimization and compare against a baseline route-based partition.

Usage (from repo root):
  conda run -n optimization python -m real_case.experiment --method LBBD --districts 3

Outputs:
  - Prints summary metrics for baseline vs optimized
  - Saves a JSON summary under lbbd_results/
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from real_case.inputs import (
    load_geodata,
    load_probability_from_population,
    load_real_odd,
    load_baseline_assignment,
    set_geodata_speeds,
)
from lib.algorithm import Partition
from lib.data import RouteData
from lib.evaluate import Evaluate


def J_function(omega):
    # Linear, consistent with compute_real_odd scaling [0,5]
    if hasattr(omega, '__len__'):
        return 0.6 * float(omega[0]) + 0.4 * float(omega[1])
    return 0.6 * float(omega)


def expand_to_square(partition: Partition, small_assign: np.ndarray, centers: list[str]) -> np.ndarray:
    # Map center list to a square N×N assignment using helper
    if small_assign.shape[1] == len(partition.short_geoid_list):
        return small_assign
    return partition._expand_assignment_matrix(small_assign, centers)


def _aggregate_prob_by_roots(assignment: np.ndarray, block_ids: List[str], center_list: List[str], prob: Dict[str, float]) -> Dict[str, float]:
    """Sum probability mass per district root given a 0/1 assignment (N x N)."""
    root_mass = {root: 0.0 for root in center_list}
    idx_of = {b: i for i, b in enumerate(block_ids)}
    for root in center_list:
        j = idx_of[root]
        # all i assigned to column j
        for i, bi in enumerate(block_ids):
            if round(assignment[i, j]) == 1:
                root_mass[root] += prob.get(bi, 0.0)
    return root_mass


def _district_blocks(assignment: np.ndarray, block_ids: List[str], root_id: str) -> List[str]:
    j = block_ids.index(root_id)
    res = []
    for i, bi in enumerate(block_ids):
        if round(assignment[i, j]) == 1:
            res.append(bi)
    return res


def evaluate_design(partition: Partition, assignment: np.ndarray, prob_dict, Omega_dict):
    # Determine a reasonable depot for this partition
    # First, identify root ids from current assignment
    block_ids = partition.short_geoid_list
    N = len(block_ids)
    roots = [block_ids[i] for i in range(N) if round(assignment[i, i]) == 1]
    if not roots:
        # pick columns with any assigned rows as roots
        cols = np.where(assignment.sum(axis=0) > 0)[0]
        roots = [block_ids[i] for i in cols]
    depot_id = partition.optimize_depot_location(assignment, roots, prob_dict)
    obj, district_info = partition.evaluate_partition_objective(
        assignment, depot_id, prob_dict,
        Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120,
        Omega_dict=Omega_dict, J_function=J_function
    )
    # Summaries
    K_sum = float(sum(info[2] for info in district_info))
    F_sum = float(sum(info[3] for info in district_info))
    return {
        'obj': float(obj),
        'depot': depot_id,
        'K_sum': K_sum,
        'F_sum': F_sum,
        'num_districts': len(district_info)
    }


def _evaluate_fixed_route_full(geodata, prob: Dict[str, float]) -> Dict:
    """Evaluate Fixed-Route (FR) rider metrics using nearest stops and SimPy wait/transit simulation.
    Provider metrics are left as None (route operations treated as sunk in this framework).
    Returns a dict with expected and worst-P rider metrics.
    """
    rd = RouteData('hct_routes.json', geodata)
    ev = Evaluate(rd, geodata)

    # Nearest stop per node for walk times
    nearest = rd.find_nearest_stops()
    nodes = list(geodata.short_geoid_list)

    # Expected walk (m/s speed 1.4)
    walking_speed = 1.4
    expected_walk_time = 0.0
    max_walk_time = 0.0
    for n in nodes:
        d = float(nearest[n]['distance'])  # meters
        t = d / walking_speed
        p = float(prob.get(n, 0.0))
        expected_walk_time += p * t
        max_walk_time = max(max_walk_time, t)

    # Worst-P walk using Wasserstein-ball maximization
    try:
        worst_prob, _ = ev.get_fixed_route_worst_distribution(prob)
        worstP_walk_time = sum((worst_prob[n] * float(nearest[n]['distance']) / walking_speed) for n in nodes)
    except Exception:
        worst_prob = None
        worstP_walk_time = None

    # Expected wait/transit using SimPy
    try:
        wait_summary, transit_summary = ev.simulate_wait_and_transit_fixed_route_simpy(prob)
        # Mass-weighted averages across routes
        total_mass = sum(v[0] for v in wait_summary.values()) or 1.0
        expected_wait_time = sum(m * w for (m, w) in wait_summary.values()) / total_mass
        expected_transit_time = sum(m * tr for (m, tr) in transit_summary.values()) / total_mass
    except Exception:
        expected_wait_time = None
        expected_transit_time = None

    return {
        'rider': {
            'expected': {
                'walk_s': expected_walk_time,
                'wait_s': expected_wait_time,
                'transit_s': expected_transit_time,
            },
            'worstP': {
                'walk_s': worstP_walk_time,
                'wait_s': expected_wait_time,   # sunk for FR
                'transit_s': expected_transit_time,  # sunk for FR
            },
            'worst_rider': {
                'walk_s': max_walk_time,
            }
        },
        'provider': {
            'expected': None,
            'worst': None,
        }
    }


def _evaluate_tsp_partition(
    geodata,
    assignment: np.ndarray,
    center_list: List[str],
    prob: Dict[str, float],
    Omega: Dict[str, np.ndarray],
    depot_id: str,
    overall_arrival_rate: float,
    use_evaluate: bool = True,
    worst_case: bool = True,
    max_dispatch_interval: float = 24.0,
    override_T_by_partition: Dict[str, float] | None = None,
) -> Dict:
    """
    Evaluate TSP last‑mile metrics for a given partition.

    - Rider metrics: mean wait per rider (T/2), mean transit per rider based on evaluate's
      mean_transit_distance_per_interval divided by expected riders per interval and wv.
    - Provider metrics: linehaul distance rate (2*dist(depot,root)/T) [km/h],
      ODD (J(max Omega in district)/T), travel distance rate (amt_transit_distance) [km/h].
    Returns dict with 'per_district' and 'aggregate' for expected vs worst-P.
    """
    # Constants for conversion
    # Use km/h consistently
    wv = getattr(geodata, 'wv_kmh', getattr(geodata, 'wv', 10.0) * 1.60934)
    # build evaluator only if needed
    rd = RouteData('hct_routes.json', geodata)
    ev = Evaluate(rd, geodata)

    # Ensure square assignment and centers
    block_ids = list(geodata.short_geoid_list)
    if set(center_list) != set(block_ids):
        # expand small to square for consistency
        # center_list already maps to root ids present in columns of small assignment
        pass

    # Mass per district (for aggregation); this is with nominal prob
    mass_by_root = _aggregate_prob_by_roots(assignment, block_ids, center_list, prob)

    # Get evaluate outputs per district (worst-case or expected)
    eval_dict = ev.evaluate_tsp_mode(
        prob_dict=prob,
        node_assignment=assignment,
        center_list=center_list,
        unit_wait_cost=1.0,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=worst_case,
        max_dipatch_interval=max_dispatch_interval,
    )

    per_district = {}
    # Build provider/rider metrics using dispatch intervals
    for root in center_list:
        dres = eval_dict[root]
        T = float(dres['dispatch_interval'])
        if override_T_by_partition and root in override_T_by_partition:
            T = float(override_T_by_partition[root])

        # Rider: wait per rider (hours)
        wait_per_rider_h = T / 2.0

        # Rider: transit per rider (hours)
        #  - service leg: (km per interval) / (riders per interval) / (km/h)
        dist_interval = float(dres['mean_transit_distance_per_interval'])  # km per interval (aggregate)
        district_mass = mass_by_root.get(root, 0.0)
        riders_per_interval = overall_arrival_rate * district_mass * T
        if riders_per_interval > 0:
            transit_time_per_rider_h = (dist_interval / riders_per_interval) / (wv)
        else:
            transit_time_per_rider_h = 0.0

        # Provider: linehaul + ODD + travel
        #  - linehaul: use shortest depot→district distance, robust to whether roots have been relocated
        assigned_blocks = _district_blocks(assignment, block_ids, root)
        if assigned_blocks:
            dr_min = min(geodata.get_dist(depot_id, b) for b in assigned_blocks)  # km
        else:
            dr_min = geodata.get_dist(depot_id, root)  # km
        # Distance rate (km/h) for linehaul over time: 2*distance per cycle divided by interval length T
        linehaul_kmph = (2.0 * dr_min) / T

        #  - ODD cost: J(max Omega in district)/T treated as minutes proxy (already a scalar). Keep as is.
        omega_vec = np.zeros(2)
        for b in assigned_blocks:
            if b in Omega:
                ov = Omega[b]
                # elementwise max
                omega_vec = np.maximum(omega_vec, ov)
        odd_cost = float(J_function(omega_vec)) / T

        #  - travel: amt_transit_distance is already km per hour
        amt_transit_distance = float(dres['amt_transit_distance'])  # km per hour
        travel_kmph = amt_transit_distance

        #  - on-board linehaul leg per rider: half of the linehaul time (outbound)
        transit_time_per_rider_h += (dr_min / wv)

        per_district[root] = {
            'dispatch_interval_h': T,
            'rider_wait_h': wait_per_rider_h,
            'rider_transit_h': transit_time_per_rider_h,
            'provider_linehaul_kmph': linehaul_kmph,
            'provider_odd_cost': odd_cost,
            'provider_travel_kmph': travel_kmph,
            'mass': district_mass,
        }

    # Aggregate by mass (riders) and sum provider components
    tot_mass = sum(per_district[r]['mass'] for r in center_list) or 1.0
    rider_wait = sum(per_district[r]['rider_wait_h'] * per_district[r]['mass'] for r in center_list) / tot_mass
    rider_transit = sum(per_district[r]['rider_transit_h'] * per_district[r]['mass'] for r in center_list) / tot_mass
    provider_linehaul = sum(per_district[r]['provider_linehaul_kmph'] for r in center_list)
    provider_odd = sum(per_district[r]['provider_odd_cost'] for r in center_list)
    provider_travel = sum(per_district[r]['provider_travel_kmph'] for r in center_list)

    return {
        'per_district': per_district,
        'aggregate': {
            'rider_wait_h': rider_wait,
            'rider_transit_h': rider_transit,
            'provider_linehaul_kmph': provider_linehaul,
            'provider_odd_cost': provider_odd,
            'provider_travel_kmph': provider_travel,
        }
    }


def run(method: str, num_districts: int, visualize_baseline: bool = False, max_iters: int = 200,
        overall_arrival_rate: float = 1000.0, max_dispatch_cap: float = 1.5):
    geodata = load_geodata()
    # Set speeds (mph) from routes and walking baseline
    wv_kmh, wr_kmh = set_geodata_speeds(geodata, 'hct_routes.json', walk_kmh=5.0)
    prob = load_probability_from_population(geodata)
    Omega = load_real_odd()

    # Baseline assignment from routes
    base_assign_small, centers = load_baseline_assignment(geodata, visualize=visualize_baseline)

    part = Partition(geodata, num_districts=num_districts, prob_dict=prob, epsilon=0.1)
    base_assign = expand_to_square(part, base_assign_small, centers)
    base_metrics = evaluate_design(part, base_assign, prob, Omega)

    # Optimized design
    district_info = None  # ensure defined for both branches
    if method.upper() == 'LBBD':
        result = part.benders_decomposition(
            max_iterations=50, tolerance=1e-3, verbose=True,
            Omega_dict=Omega, J_function=J_function,
            Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120
        )
        # Support both dict and tuple legacy returns
        if isinstance(result, dict):
            opt_assign = result.get('best_partition')
            opt_depot = result.get('best_depot')
            district_info = result.get('best_district_info')
        else:
            # legacy: (best_partition, best_cost, history)
            opt_assign = result[0]
            opt_depot = None
            district_info = None
        if opt_assign is None:
            raise RuntimeError('LBBD did not return a partition')
        if opt_depot is None:
            # fallback evaluation to compute depot
            metrics = evaluate_design(part, opt_assign, prob, Omega)
            opt_depot = metrics['depot']
        else:
            obj, district_info2 = part.evaluate_partition_objective(
                opt_assign, opt_depot, prob, 1.0, 1.0, 10.0, 0.7120, Omega, J_function
            )
            district_info = district_info or district_info2
            metrics = {
                'obj': float(obj),
                'depot': opt_depot,
                'K_sum': float(sum(info[2] for info in district_info)) if district_info else None,
                'F_sum': float(sum(info[3] for info in district_info)) if district_info else None,
                'num_districts': len(district_info) if district_info else None
            }

        # Relocate optimized roots to be closest-to-depot to align linehaul with root distance
        try:
            opt_assign, _opt_roots_rel = part.relocate_roots_to_depot_closest(opt_assign, opt_depot)
            # Recompute district_info with relocated roots so T* maps correctly by root id
            obj_post, district_info_post = part.evaluate_partition_objective(
                opt_assign, opt_depot, prob, 1.0, 1.0, 10.0, 0.7120, Omega, J_function
            )
            district_info = district_info_post
            metrics['obj'] = float(obj_post)
        except Exception:
            pass
    else:
        depot, roots, assign, obj_val, rs_district_info = part.random_search(
            max_iters=max_iters, prob_dict=prob, Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega, J_function=J_function
        )
        opt_assign = assign
        district_info = rs_district_info
        metrics = {
            'obj': float(obj_val),
            'depot': depot,
            'K_sum': None,
            'F_sum': None,
            'num_districts': int((assign.sum(axis=0) > 0).sum())
        }

    # ---------- Extended Evaluation Scenarios ----------
    # 1) Fixed-route under original (route-based) partition
    fr_eval = _evaluate_fixed_route_full(geodata, prob)

    # Helper: extract center list for partitions (columns with any assignment)
    block_ids = list(geodata.short_geoid_list)
    base_roots = [block_ids[i] for i in range(len(block_ids)) if round(base_assign[i, i]) == 1]
    if not base_roots:
        cols = np.where(base_assign.sum(axis=0) > 0)[0]
        base_roots = [block_ids[i] for i in cols]

    # Compute depot for baseline partition
    base_depot = part.optimize_depot_location(base_assign, base_roots, prob)

    # 2) TSP under original partition (unconstrained and constrained)
    #    Relocate baseline roots to be depot-closest so dist(depot, root) equals the
    #    shortest distance to the district, per modeling requirement.
    try:
        base_assign, base_roots = part.relocate_roots_to_depot_closest(base_assign, base_depot)
    except Exception:
        pass
    # Relocate baseline roots to be depot-closest to align root IDs and distances
    try:
        base_assign, base_roots = part.relocate_roots_to_depot_closest(base_assign, base_depot)
    except Exception:
        pass

    # Compute realistic T* for baseline partition AFTER relocation so root IDs match
    try:
        _obj_base, _district_info_base = part.evaluate_partition_objective(
            base_assign, base_depot, prob,
            Lambda=1.0, wr=1.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega, J_function=J_function
        )
        T_override_base = {str(info[1]): float(info[4]) for info in _district_info_base}
    except Exception:
        T_override_base = None

    tsp_base_unconstrained = _evaluate_tsp_partition(
        geodata, base_assign, base_roots, prob, Omega, base_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=24.0,
        override_T_by_partition=T_override_base,
    )
    tsp_base_constrained = _evaluate_tsp_partition(
        geodata, base_assign, base_roots, prob, Omega, base_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=max_dispatch_cap,
        override_T_by_partition=T_override_base,
    )

    # 3) TSP under optimal partition (use T* returned by optimization if available; unconstrained + constrained)
    opt_roots = [block_ids[i] for i in range(len(block_ids)) if round(opt_assign[i, i]) == 1]
    if not opt_roots:
        cols = np.where(opt_assign.sum(axis=0) > 0)[0]
        opt_roots = [block_ids[i] for i in cols]

    # Map T* per root from district_info, if available (hours)
    T_override = None
    if district_info:
        try:
            # district_info entries: [cost, root, K_i, F_i, T_star, ...]
            T_override = {str(info[1]): float(info[4]) for info in district_info}
        except Exception:
            T_override = None

    opt_depot = metrics['depot']
    tsp_opt_unconstrained = _evaluate_tsp_partition(
        geodata, opt_assign, opt_roots, prob, Omega, opt_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=24.0,
        override_T_by_partition=T_override,
    )
    tsp_opt_constrained = _evaluate_tsp_partition(
        geodata, opt_assign, opt_roots, prob, Omega, opt_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=max_dispatch_cap,
        override_T_by_partition=T_override,
    )

    # Report & save
    summary = {
        'method': method,
        'num_districts': num_districts,
        'baseline': base_metrics,
        'optimized': metrics,
        'improvement': float(base_metrics['obj'] - metrics['obj']),
        'evaluations': {
            'FR_original': fr_eval,
            'TSP_original_unconstrained': tsp_base_unconstrained,
            'TSP_original_constrained': tsp_base_constrained,
            'TSP_opt_unconstrained': tsp_opt_unconstrained,
            'TSP_opt_constrained': tsp_opt_constrained,
        }
    }
    print(json.dumps(summary, indent=2))

    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join('results', f'real_case_summary_{ts}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Also emit a compact CSV-like table for quick comparison
    table_lines = []
    table_lines.append('Design,Perspective,Metric,WorstP_h,Expected_h,ProviderWorst_kmph,ProviderExpected_kmph')
    # FR
    fr_r = fr_eval['rider']
    fr_p = fr_eval['provider']
    # Convert FR rider seconds to hours
    fr_walk_worstP_h = (fr_r['worstP']['walk_s'] or 0.0) / 3600.0
    fr_walk_exp_h = (fr_r['expected']['walk_s'] or 0.0) / 3600.0
    fr_wait_exp_h = (fr_r['expected']['wait_s'] or 0.0) / 3600.0
    fr_transit_exp_h = (fr_r['expected']['transit_s'] or 0.0) / 3600.0
    table_lines.append(f"P0 FR,Rider,walk,{fr_walk_worstP_h},{fr_walk_exp_h},,")
    table_lines.append(f"P0 FR,Rider,wait,,{fr_wait_exp_h},,")
    table_lines.append(f"P0 FR,Rider,transit,,{fr_transit_exp_h},,")
    # TSP original
    for tag, res in [('P0 TSP unconstrained', tsp_base_unconstrained), ('P0 TSP constrained', tsp_base_constrained)]:
        agg = res['aggregate']
        table_lines.append(f"{tag},Rider,wait,{agg['rider_wait_h']},,{agg['provider_linehaul_kmph']+agg['provider_travel_kmph']},{agg['provider_linehaul_kmph']+agg['provider_travel_kmph']}")
        table_lines.append(f"{tag},Rider,transit,{agg['rider_transit_h']},,,")
    # TSP optimal
    for tag, res in [('P* TSP unconstrained', tsp_opt_unconstrained), ('P* TSP constrained', tsp_opt_constrained)]:
        agg = res['aggregate']
        table_lines.append(f"{tag},Rider,wait,{agg['rider_wait_h']},,{agg['provider_linehaul_kmph']+agg['provider_travel_kmph']},{agg['provider_linehaul_kmph']+agg['provider_travel_kmph']}")
        table_lines.append(f"{tag},Rider,transit,{agg['rider_transit_h']},,,")

    table_path = os.path.join('lbbd_results', f'real_case_table_{ts}.csv')
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))
    print(f"Saved summary to {summary_path} and table to {table_path}")


def main():
    ap = argparse.ArgumentParser(description='Run real-case optimization experiment and compare to baseline')
    ap.add_argument('--method', choices=['LBBD', 'RS'], default='LBBD')
    ap.add_argument('--districts', type=int, default=3)
    ap.add_argument('--visualize-baseline', action='store_true')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--arrival-rate', type=float, default=100.0, help='Overall arrival rate per hour for TSP evaluation')
    ap.add_argument('--Tmax', type=float, default=1.5, help='Constrained dispatch subinterval cap (hours)')
    args = ap.parse_args()
    run(args.method, args.districts, args.visualize_baseline, args.iters, overall_arrival_rate=args.arrival_rate, max_dispatch_cap=args.Tmax)


if __name__ == '__main__':
    main()
