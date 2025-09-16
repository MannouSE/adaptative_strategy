import random
from types import SimpleNamespace
from evrp.data import Problem
from evrp.solution import generate_initial_solution, quick_repair
from evrp.costs import full_cost
from . import heuristics
from .elite import update_elite_archive, cluster_elite_archive
from .q_learning import get_best_action, update as q_update, decay_epsilon

ACTIONS = ['H1','H2','H3','H4']

def initialize_algorithm(problem: Problem, pop_size: int, eps_start: float, rng):
    P = []
    for _ in range(pop_size):
        sol = generate_initial_solution(problem)
        sol = quick_repair(sol, problem)
        P.append(sol)
    elite = {}
    centroids = []
    best_cost = float('inf'); best_sol = None
    Q = {}; state = (0, 'start'); eps = eps_start
    return P, elite, centroids, best_cost, best_sol, Q, state, eps

def main_optimization(problem: Problem, cfg: SimpleNamespace, rng: random.Random):
    P, elite, centroids, best_c, best_s, Q, state, eps = initialize_algorithm(problem, cfg.pop_size, cfg.eps_start, rng)

    for gen in range(cfg.max_gens):
        # Step 1: fitness
        fitness = [1/(1+full_cost(s, problem)) for s in P]

        # Step 2: mating pool
        M = []
        for _ in range(len(P)):
            idxs = rng.sample(range(len(P)), cfg.tournament_size)
            winner = max(idxs, key=lambda i: fitness[i])
            M.append(P[winner])

        # Step 3: action (epsilon-greedy)
        action = rng.choice(ACTIONS) if rng.random() < eps else get_best_action(Q, state, ACTIONS)

        # Step 4: offspring
        P_new = []
        for parent in M:
            if action == 'H1':
                child = heuristics.heuristic_h1_full_hierarchical(parent, elite, problem, rng)
            elif action == 'H2':
                child = heuristics.heuristic_h2_selective_ll(parent, elite, problem, rng)
            elif action == 'H3':
                child = heuristics.heuristic_h3_relaxed_ll(parent, problem, rng)
            else:
                child = heuristics.heuristic_h4_similarity_based(parent, cluster_elite_archive(elite), elite, problem, rng)
            child = quick_repair(child, problem)
            P_new.append(child)

        # Step 5: best offspring
        best_off = min(P_new, key=lambda s: full_cost(s, problem))
        best_off_cost = full_cost(best_off, problem)

        # Step 6: reward + elite update
        reward = (best_c - best_off_cost) if best_c < float('inf') else 0.0
        improved = best_off_cost < best_c
        if improved:
            best_c, best_s = best_off_cost, best_off
            update_elite_archive(elite, best_off, best_off_cost)
            centroids = cluster_elite_archive(elite)

        # Step 7: Q-learning update
        next_state = (1 if improved else 0, action)
        q_update(Q, state, action, reward, next_state, cfg.alpha, cfg.gamma, ACTIONS)

        # Step 8: survivor selection (μ+λ)
        combined = P + P_new
        combined.sort(key=lambda s: full_cost(s, problem))
        P = combined[:cfg.pop_size]

        # Step 9: epsilon decay
        eps = decay_epsilon(eps, cfg.eps_min, cfg.decay)
        state = next_state

        if gen % 50 == 0:
            print(f"[gen {gen}] best={best_c:.2f} eps={eps:.3f} act={action}")

    return best_s, best_c
