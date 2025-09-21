import random
from evrp.costs import full_cost
from evrp.solution import hash_solution
from evrp.operators import apply_ul_operator, apply_ul_operator_guided
from evrp.elite import update_elite_archive, cluster_elite_archive, select_centroid, find_nearest_centroid

def solve_ll_exact(sol_or_route, problem, rng=None):
    """
    LL objective (charge to full at a station):
      min ∑ d_ib*y_ibk + ∑ w_bk*y_ibk + ∑ r_bk*(Bmax - b_ik)*y_ibk
    Subject to 0 ≤ b_ik ≤ Bmax and b propagation along the fixed UL route.
    Works for one route or a full solution. Returns (ok, same_input, ll_cost_total).
    """
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = problem.energy_consumption or 1e-9
    stations = tuple(problem.stations or ())
    ev_range = BMAX / alpha  # max km on full battery

    # safe maps (allow empty)
    detour_km   = getattr(problem, "station_detour_km", {}) or {}
    price_map   = getattr(problem, "station_energy_price", {}) or {}
    wait_cost   = getattr(problem, "station_wait_cost", {}) or {}
    price_def   = getattr(problem, "energy_cost", 0.0) or 0.0   # $/kWh fallback
    wait_def    = getattr(problem, "waiting_cost", 0.0) or 0.0  # $/visit fallback

    # prune: only consider K nearest, reachable stations from i,
    # and only those that can reach j after a full charge
    K = getattr(problem, "k_nearest_stations", 5)
    _cand_cache = {}  # i -> list of top-K station ids by (D[i][b] + δ)

    def candidate_stations(i, j):
        lst = _cand_cache.get(i)
        if lst is None:
            lst = sorted(stations, key=lambda b: D[i][b] + detour_km.get(b, 0.0))[:K]
            _cand_cache[i] = lst
        # filter by reachability on *full* battery for both legs
        res = []
        for b in lst:
            if D[i][b] + detour_km.get(b, 0.0) <= ev_range and D[b][j] <= ev_range:
                res.append(b)
        return res

    def _one_route(route):
        if not route or len(route) < 2:
            return True, 0.0

        soc = (getattr(problem, "init_soc_ratio", 1.0) or 1.0) * BMAX
        ll_cost = 0.0

        for t in range(len(route) - 1):
            i, j = route[t], route[t + 1]
            need_direct = alpha * D[i][j]

            # BIG speedup: if direct is feasible, take it and skip station loop entirely
            if soc >= need_direct:
                soc -= need_direct
                continue

            best_cost = float('inf')
            best_b = None

            # Try only a few good stations
            for b in candidate_stations(i, j):
                # reach b from i on current soc
                need_ib = alpha * (D[i][b] + detour_km.get(b, 0.0))
                if need_ib > soc:
                    continue

                soc_arr_b = soc - need_ib
                energy_to_full = max(0.0, BMAX - soc_arr_b)

                # after full charge, must reach j
                if BMAX - alpha * D[b][j] < -1e-9:
                    continue

                detour_term = D[i][b] + detour_km.get(b, 0.0)    # d_ib (+ δ if any)
                wbk = wait_cost.get(b, wait_def)                 # $/visit
                rbk = price_map.get(b, price_def)                # $/kWh
                cand = detour_term + wbk + rbk * energy_to_full

                if cand < best_cost:
                    best_cost = cand
                    best_b = b

            if best_cost == float('inf'):
                # no feasible way to continue
                return False, float('inf')

            # apply stop at best_b (charge to full)
            soc -= alpha * (D[i][best_b] + detour_km.get(best_b, 0.0))
            soc = BMAX
            soc -= alpha * D[best_b][j]
            ll_cost += best_cost

            # bounds check
            if soc < -1e-9 or soc > BMAX + 1e-9:
                return False, float('inf')

        return True, ll_cost

    # solution vs route
    if sol_or_route and isinstance(sol_or_route[0], list):
        total = 0.0
        for r in sol_or_route:
            ok, c = _one_route(r)
            if not ok:
                return False, sol_or_route, float('inf')
            total += c
        return True, sol_or_route, total
    else:
        ok, c = _one_route(sol_or_route)
        return ok, sol_or_route, c
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
def is_promising_ul(sol_or_route, problem) -> bool:
    """
    Cheap pre-check: for every arc i->j in the route(s), require that either
    - i->j is reachable directly on a full battery, OR
    - there exists a station b such that i->b and b->j are each reachable on a full battery.
    This is a necessary (not sufficient) condition, so it’s a good quick filter before LL.
    """
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    alpha = problem.energy_consumption or 1e-9
    max_leg_km = BMAX / alpha

    stations = list(problem.stations or [])
    detour = getattr(problem, "station_detour_km", {}) or {}

    def arc_ok(i, j) -> bool:
        # direct reachability
        if D[i][j] <= max_leg_km:
            return True
        # try a single station as an intermediate (detour km optional)
        for b in stations:
            if (D[i][b] + detour.get(b, 0.0) <= max_leg_km) and (D[b][j] <= max_leg_km):
                return True
        return False

    # Handle single route vs whole solution
    if sol_or_route and isinstance(sol_or_route[0], list):
        for route in sol_or_route:
            for t in range(len(route) - 1):
                if not arc_ok(route[t], route[t + 1]):
                    return False
        return True
    else:
        route = sol_or_route
        for t in range(len(route) - 1):
            if not arc_ok(route[t], route[t + 1]):
                return False
        return True