# (you can remove the sys.path insert now)

from evrp.costs import calculate_travel_cost, full_cost
import argparse, random
from types import SimpleNamespace
from evrp.data import load_evrp, apply_defaults
from evrp.optimize import main_optimization
from evrp.generators import decorate_with_pevrp_params

def describe_solution(problem, sol):
    print("Routes:")
    for r in sol:
        print("  ", " -> ".join(map(str, r)))
    print("Travel distance:", calculate_travel_cost(sol, problem))
    print("Total cost     :", full_cost(sol, problem))


def main():
    # assumes you already have these imports at the top of the file:
    # import argparse, random
    # from types import SimpleNamespace
    # from evrp.data import load_evrp, apply_defaults
    # from evrp.optimize import main_optimization
    # from evrp.generators import decorate_with_pevrp_params

    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="Path to .evrp instance file")
    ap.add_argument("--max-gens", type=int, default=200)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--tournament-size", type=int, default=2)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-min", type=float, default=0.1)
    ap.add_argument("--decay", type=float, default=0.995)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=8)
    # optional knobs you may want later:
    ap.add_argument("--waiting-cost", type=float, default=None, help="$/hour to monetize time (optional)")
    ap.add_argument("--energy-cost", type=float, default=None, help="fallback $/kWh (optional)")
    ap.add_argument("--charge-rate", type=float, default=None, help="fallback kW if a station lacks a rate (optional)")
    ap.add_argument("--speed", type=float, default=None, help="vehicle speed km/h (optional)")
    args = ap.parse_args()

    # Resolve instance path (use helper if you have it)
    try:
        instance_path = resolve_instance_path(args.instance)  # your helper, if defined
    except NameError:
        instance_path = args.instance

    print(f"Loading instance: {instance_path}")
    problem = load_evrp(instance_path)
    problem = apply_defaults(problem)

    # Decorate with pEVRP params (fixed 30 min charge, per-station rate/price/wait/detour)
    problem = decorate_with_pevrp_params(problem, seed=args.seed, charge_rate_kW=200.0)

    # Allow CLI to override a few globals if provided
    if args.waiting_cost is not None:
        problem.waiting_cost = args.waiting_cost
    if args.energy_cost is not None:
        problem.energy_cost = args.energy_cost
    if args.charge_rate is not None:
        problem.charge_rate = args.charge_rate
    if args.speed is not None:
        problem.speed = args.speed

    cfg = SimpleNamespace(
        max_gens=args.max_gens,
        pop_size=args.pop,
        tournament_size=args.tournament_size,
        eps_start=args.eps_start,
        eps_min=args.eps_min,
        decay=args.decay,
        alpha=args.alpha,
        gamma=args.gamma,
    )

    rng = random.Random(args.seed)
    best_sol, best_cost = main_optimization(problem, cfg, rng)

    print("=== DONE ===")
    print("Best cost:", best_cost)
    if best_sol:
        print("Best solution structure:", [len(r) for r in best_sol])

if __name__ == "__main__":
    main()
