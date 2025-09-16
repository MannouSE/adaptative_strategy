# evrp/costs.py
from .data import Problem

BIG_M_PENALTY = 10**9
CAP_PENALTY = 1e6  # optional capacity penalty for load overflow

def calculate_travel_cost(solution, problem: Problem) -> float:
    """Sum of travel distances over all arcs (optionally adds per-station detour km on arrival)."""
    D = problem.distance_matrix
    stations = set(problem.stations or [])
    total = 0.0
    for route in solution:
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            dist = D[u][v]
            if v in stations:
                dist += problem.station_detour_km.get(v, 0.0)
            total += dist
    return total

def route_load(route, problem: Problem) -> int:
    """Total demand (kg) served on a route."""
    if not problem.demands:
        return 0
    load = 0
    customers = set(problem.customers or [])
    for n in route:
        if n in customers:
            load += problem.demands.get(n, 1)
    return load

def capacity_violation(solution, problem: Problem) -> int:
    """Sum of positive overloads across routes (kg)."""
    if not problem.demands:
        return 0
    Q = problem.capacity
    viol = 0
    for r in solution:
        viol += max(0, route_load(r, problem) - Q)
    return viol

def check_energy_feasibility(solution, problem: Problem):
    BMAX = problem.energy_capacity
    alpha = problem.energy_consumption        # kWh per km
    D = problem.distance_matrix
    stations = set(problem.stations or [])
    fixed_time_h = getattr(problem, "fixed_charge_time_h", 0.5)  # 30 minutes
    wait_rate = getattr(problem, "waiting_cost", 0.0) or 0.0

    # safe maps
    detour_map      = getattr(problem, "station_detour_km", {}) or {}
    charge_rate_map = getattr(problem, "station_charge_rate", {}) or {}
    energy_price_map= getattr(problem, "station_energy_price", {}) or {}
    wait_time_map   = getattr(problem, "station_wait_time", {}) or {}
    per_visit_map   = getattr(problem, "station_wait_cost", {}) or {}

    total_energy_cost = 0.0
    total_waiting_cost = 0.0
    feasible = True

    for route in solution:
        soc = (getattr(problem, "init_soc_ratio", 1.0) or 1.0) * BMAX
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]

            dist = D[u][v]
            if v in stations:
                dist += detour_map.get(v, 0.0)
            soc -= dist * alpha
            if soc < -1e-9:
                feasible = False
                break

            if v in stations:
                r_kW = charge_rate_map.get(v, getattr(problem, "charge_rate", float("inf")))
                energy_added = min(BMAX - soc, r_kW * fixed_time_h)

                price_b = energy_price_map.get(v, getattr(problem, "energy_cost", 0.0))
                total_energy_cost += energy_added * price_b

                base_wait_h = wait_time_map.get(v, 0.0)
                total_waiting_cost += (base_wait_h + fixed_time_h) * wait_rate
                total_waiting_cost += per_visit_map.get(v, 0.0)

                soc = min(BMAX, soc + energy_added)

        if not feasible:
            break

    return feasible, total_energy_cost, total_waiting_cost
def full_cost(solution, problem: Problem) -> float:
    travel = calculate_travel_cost(solution, problem)
    feasible, e_cost, w_cost = check_energy_feasibility(solution, problem)
    penalty = 0.0 if feasible else BIG_M_PENALTY
    penalty += CAP_PENALTY * capacity_violation(solution, problem)
    cost_total= travel + e_cost + w_cost + penalty
    #print("cost total upper level",cost_total)
    return travel + e_cost + w_cost + penalty
