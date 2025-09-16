import random
from evrp.costs import full_cost
from evrp.solution import hash_solution
from evrp.operators import apply_ul_operator, apply_ul_operator_guided
from evrp.elite import update_elite_archive, cluster_elite_archive, select_centroid, find_nearest_centroid

def solve_ll_exact(sol_or_route, problem, rng=None):
    """
    Lower-level 'exact' charging with fixed 30-min per stop.
    Works on a single route [n0,...,nm] OR a full solution [[...], [...], ...].
    Returns (ok, same_input, ll_cost_total).
    """
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = problem.energy_consumption           # kWh/km
    stations = set(problem.stations or [])
    fixed = getattr(problem, "fixed_charge_time_h", 0.5)  # 0.5 h = 30 min
    wait_rate = problem.waiting_cost or 0.0

    def _ll_route_cost(route):
        # route: list[int] like [dep,...,dep]
        if not route or len(route) < 2:
            return True, 0.0
        soc = (getattr(problem, "init_soc_ratio", 1.0) or 1.0) * BMAX
        ll_cost = 0.0
        for t in range(len(route)-1):
            i, j = route[t], route[t+1]
            # Option A: go direct i->j if SoC allows
            need_direct = D[i][j] * alpha
            best_kind = 'none'
            best_b = None
            best_n = 0.0
            best_cost = 0.0 if soc >= need_direct else float('inf')

            # Option B: detour i->b (station) then b->j with fixed 30min charge
            for b in stations:
                # include extra detour km when going to station b (if you model it)
                detour_km = problem.station_detour_km.get(b, 0.0)
                need_to_b = (D[i][b] + detour_km) * alpha
                if need_to_b > soc:
                    continue
                e_arr_b = soc - need_to_b
                need_b_to_j = D[b][j] * alpha

                r_b = problem.station_charge_rate.get(b, problem.charge_rate or float("inf"))  # kW
                n = min(max(0.0, need_b_to_j - e_arr_b), BMAX - e_arr_b, r_b * fixed)          # kWh addable in 30 min
                if e_arr_b + n < need_b_to_j - 1e-9:
                    # even after 30 min you can't reach j from b
                    continue

                detour_dist_term = (D[i][b] + detour_km)  # matches your dib + δ
                base_wait_h = problem.station_wait_time.get(b, 0.0)
                # optional per-visit lump sum
                per_visit = getattr(problem, "station_wait_cost", {}).get(b, 0.0)
                wait_c = (base_wait_h + fixed) * wait_rate + per_visit
                energy_c = n * problem.station_energy_price.get(b, (problem.energy_cost or 0.0))
                cand_cost = detour_dist_term + wait_c + energy_c

                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_kind, best_b, best_n = 'charge', b, n

            if best_cost == float('inf'):
                return False, float('inf')

            # apply choice and advance SoC
            if best_kind == 'none':
                soc -= need_direct
            else:
                detour_km = problem.station_detour_km.get(best_b, 0.0)
                soc = soc - (D[i][best_b] + detour_km) * alpha
                soc = min(BMAX, soc + best_n)
                soc -= D[best_b][j] * alpha
                ll_cost += best_cost

            if soc < -1e-9:
                return False, float('inf')

        return True, ll_cost

    # Detect whether we got a full solution or a single route
    if sol_or_route and isinstance(sol_or_route[0], list):
        total = 0.0
        for r in sol_or_route:
            ok, c = _ll_route_cost(r)
            if not ok:
                return False, sol_or_route, float('inf')
            total += c
        return True, sol_or_route, total
    else:
        ok, c = _ll_route_cost(sol_or_route)
        return ok, sol_or_route, c
def is_promising_ul(sol, problem):
    # promising if no penalty (i.e., feasible) under current full_cost
    return full_cost(sol, problem) < 1e9

def heuristic_h1_full_hierarchical(parent, elite, problem, rng=random):
    ok, ll_sol, ll_cost = solve_ll_exact(parent, problem, rng)
    if ok:
        update_elite_archive(elite, ll_sol, full_cost(ll_sol, problem))
        centroids = cluster_elite_archive(elite)
        pick = select_centroid(centroids, rng)
        if pick is not None:
            # mutate the picked centroid to avoid cloning
            return apply_ul_operator(pick, problem, rng)
    return apply_ul_operator(parent, problem, rng)

def heuristic_h2_selective_ll(parent, elite, problem, rng=random):
    if is_promising_ul(parent, problem):
        ok, ll_sol, ll_cost = solve_ll_exact(parent, problem, rng)
        if ok:
            update_elite_archive(elite, ll_sol, full_cost(ll_sol, problem))
    return apply_ul_operator(parent, problem, rng)

def heuristic_h3_relaxed_ll(parent, problem, rng=random):
    return apply_ul_operator(parent, problem, rng)

def heuristic_h4_similarity_based(parent, centroids, elite, problem, rng=random):
    if not centroids:
        return apply_ul_operator(parent, problem, rng)
    near = find_nearest_centroid(parent, centroids, problem, rng)
    if near is not None:
        # start from the nearest centroid and mutate
        return apply_ul_operator(near, problem, rng)
    return apply_ul_operator(parent, problem, rng)
