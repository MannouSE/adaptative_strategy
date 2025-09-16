from typing import Dict, Tuple, List
from evrp.solution import hash_solution
from evrp.costs import full_cost

# elite: Dict[int, Tuple[solution, cost]]

def update_elite_archive(elite: Dict[int, Tuple], solution, cost: float, max_size: int = 100):
    h = hash_solution(solution)
    elite[h] = (solution, cost)
    if len(elite) > max_size:
        # drop worst by cost
        worst_h = max(elite.keys(), key=lambda k: elite[k][1])
        elite.pop(worst_h, None)

def cluster_elite_archive(elite: Dict[int, Tuple], k: int = 3) -> List:
    if not elite:
        return []
    # "centroids" = top-k best solutions by cost (simple & dependency-free)
    top = sorted((v for v in elite.values()), key=lambda t: t[1])[:k]
    return [sol for (sol, c) in top]

def select_centroid(centroids: List, rng):
    if not centroids:
        return None
    return rng.choice(centroids)

def find_nearest_centroid(parent, centroids: List, problem, rng):
    # pick centroid with closest travel cost (very rough similarity)
    if not centroids:
        return None
    def travel(sol): return full_cost(sol, problem)  # includes penalty; fine for a stub
    target = travel(parent)
    return min(centroids, key=lambda c: abs(travel(c) - target))
