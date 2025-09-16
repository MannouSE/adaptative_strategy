from typing import List
from evrp.data import Problem

def clone_solution(sol): return [r[:] for r in sol]
def hash_solution(sol) -> int: return hash(tuple(tuple(r) for r in sol))

def generate_initial_solution(problem: Problem) -> List[List[int]]:
    import random
    customers = problem.customers[:]
    random.shuffle(customers)
    routes = [[] for _ in range(problem.vehicles)]
    for i, c in enumerate(customers):
        routes[i % problem.vehicles].append(c)
    return [[problem.depot] + r + [problem.depot] for r in routes]

def _insert_stations_until_feasible(route: List[int], problem: Problem) -> List[int]:
    """Greedy: when u->v would make SoC < 0, insert the best reachable station between them.
       'Best' = minimal extra distance. Charge fully at each station."""
    D = problem.distance_matrix
    BMAX = problem.energy_capacity
    rate = problem.energy_consumption
    stations = set(problem.stations or [])

    r = route[:]  # work on a copy
    soc = BMAX
    i = 0
    while i < len(r) - 1:
        u, v = r[i], r[i+1]
        dist_uv = D[u][v]
        need = dist_uv * rate

        # if we can go u->v, do it
        if soc >= need:
            soc -= need
            # full charge if next node is a station
            if v in stations:
                # full charge (you can change to partial later)
                soc = BMAX
            i += 1
            continue

        # otherwise: we need a station between u and v
        # pick any station s that is REACHABLE from u with current soc
        reachable = []
        for s in stations:
            need_us = D[u][s] * rate
            if need_us <= soc and s not in (u, v):
                extra = D[u][s] + D[s][v] - dist_uv
                reachable.append((extra, s))

        if not reachable:
            # no station reachable with current SoC => give up gracefully (leave route as-is)
            # You can add backtracking here if you hit this on real data.
            return r

        # choose station that adds least extra distance
        _, s_best = min(reachable, key=lambda t: t[0])

        # insert s between u and v
        r.insert(i+1, s_best)
        # travel u->s and fully charge there
        soc -= D[u][s_best] * rate
        soc = BMAX
        # next loop iteration will consider s->v
    return r

def quick_repair(solution, problem: Problem):
    repaired = []
    for r in solution:
        route = r[:]
        if not route or route[0] != problem.depot:
            route = [problem.depot] + route
        if route[-1] != problem.depot:
            route = route + [problem.depot]
        # ensure SoC feasibility by inserting stations as needed
        route = _insert_stations_until_feasible(route, problem)
        repaired.append(route)
    return repaired
