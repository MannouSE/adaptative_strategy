# evrp/costs.py
from .data import Problem

BIG_M_PENALTY = 10**9
CAP_PENALTY = 1e6  # optional capacity penalty for load overflow

def calculate_travel_cost(solution, problem: Problem) -> float:
    D = problem.distance_matrix
    total = 0.0
    for route in solution:
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            total += D[u][v]      # ❗ no station detour here
    return total

def route_load(route, problem: Problem) -> int:
    if not problem.demands: return 0
    cust = set(problem.customers or [])
    return sum(problem.demands.get(n, 1) for n in route if n in cust)

def capacity_violation(solution, problem: Problem) -> int:
    if not problem.demands: return 0
    Q = problem.capacity
    return sum(max(0, route_load(r, problem) - Q) for r in solution)

def full_cost(solution, problem):
    from .heuristics import solve_ll_exact
    base = calculate_travel_cost(solution, problem)
    ok, _, ll_cost = solve_ll_exact(solution, problem)   # detour + wbk + r*(Bmax - b_i)
    penalty = 0.0 if ok else BIG_M_PENALTY
    penalty += CAP_PENALTY * capacity_violation(solution, problem)
    return base + ll_cost + penalty