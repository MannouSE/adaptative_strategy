import random
from typing import List, Tuple
from .solution import clone_solution
from .costs import full_cost

def _valid_client_positions(route_len: int) -> range:
    # exclude depot at index 0 and last
    return range(1, max(1, route_len - 1))

def _two_opt_on_route(route: List[int], rng=random) -> List[int]:
    if len(route) < 4:  # need at least depot, a, b, depot
        return route[:]
    i = rng.randint(1, len(route) - 3)
    j = rng.randint(i + 1, len(route) - 2)
    return route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]

def _relocate(solution: List[List[int]], rng=random) -> List[List[int]]:
    child = clone_solution(solution)
    # pick source route with a client
    src_idxs = [ri for ri, r in enumerate(child) if len(r) > 3]
    if not src_idxs:
        return child
    ri = rng.choice(src_idxs)
    src = child[ri]
    pos_i = rng.choice(list(_valid_client_positions(len(src))))
    node = src.pop(pos_i)
    # pick dest route
    dest_idxs = list(range(len(child)))
    rj = rng.choice(dest_idxs)
    dest = child[rj]
    insert_pos = rng.randint(1, len(dest) - 1)  # before last depot
    dest.insert(insert_pos, node)
    return child

def _swap(solution: List[List[int]], rng=random) -> List[List[int]]:
    child = clone_solution(solution)
    # choose two routes (can be same for intra-route swap)
    if not child:
        return child
    a = rng.randrange(len(child))
    b = rng.randrange(len(child))
    ra, rb = child[a], child[b]
    if len(ra) < 3 or len(rb) < 3:
        return child
    i = rng.choice(list(_valid_client_positions(len(ra))))
    j = rng.choice(list(_valid_client_positions(len(rb))))
    ra[i], rb[j] = rb[j], ra[i]
    return child

def apply_ul_operator(parent: List[List[int]], problem, rng=random, n_candidates: int = 8):
    """
    Try a small neighborhood (2-opt / relocate / swap) and return the best candidate
    that improves (or ties) the parent's cost.
    """
    parent_cost = full_cost(parent, problem)
    best = None
    best_cost = float('inf')

    moves = (_two_opt_on_route, _relocate, _swap)

    for _ in range(n_candidates):
        move = rng.choice(moves)
        if move is _two_opt_on_route:
            # pick a route to 2-opt
            ridxs = [i for i, r in enumerate(parent) if len(r) > 3]
            if not ridxs:
                continue
            ridx = rng.choice(ridxs)
            cand = clone_solution(parent)
            cand[ridx] = _two_opt_on_route(cand[ridx], rng)
        else:
            cand = move(parent, rng)

        c = full_cost(cand, problem)
        if c < best_cost:
            best, best_cost = cand, c

    if best is not None and best_cost <= parent_cost:
        return best
    # small chance to accept the best candidate even if slightly worse (diversification)
    if best is not None and best_cost < parent_cost * 1.002:
        return best
    return parent

def apply_ul_operator_guided(parent, ll_hint_cost, problem, rng=random, n_candidates: int = 8):
    # For now we just reuse the improved neighborhood search.
    # You can bias selection using ll_hint_cost later if you compute it.
    return apply_ul_operator(parent, problem, rng, n_candidates)
