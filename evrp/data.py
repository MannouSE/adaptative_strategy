from __future__ import annotations
from dataclasses import dataclass, field   # ← field is required for default_factory
from typing import Optional
from math import sqrt
from typing import List, Tuple, Dict


@dataclass
class Problem:
    # Basic instance info
    name: str = "unnamed"
    vehicles: int = 1
    capacity: int = 1000                       # kg payload
    depot: int = 1
    customers: List[int] = field(default_factory=list)
    stations: List[int] = field(default_factory=list)

    has_time_windows: bool = False
    ready_time: Dict[int, float] = field(default_factory=dict)
    due_time: Dict[int, float] = field(default_factory=dict)
    service_duration: Dict[int, float] = field(default_factory=dict)

    # Geometry / distances
    coords: List[tuple] = field(default_factory=list)
    distance_matrix: List[List[float]] = field(default_factory=list)

    # Energy model
    energy_capacity: float = 100.0             # kWh (Bmax)
    energy_consumption: float = 1/6.0          # kWh/km (1/gamma)
    init_soc_ratio: float = 1.0                # α0
    speed: Optional[float] = None              # km/h (optional)

    # Costs & charging
    waiting_cost: float = 0.0                  # $/hour
    energy_cost: float = 0.0                   # fallback $/kWh
    charge_rate: Optional[float] = None        # fallback kW
    fixed_charge_time_h: float = 0.5           # 30 minutes per recharge (fixed)

    # Demands (for capacity), optional
    demands: Dict[int, int] = field(default_factory=dict)

    # Per-station parameters (all optional; filled by decorator or generator)
    station_charge_rate: Dict[int, float] = field(default_factory=dict)  # kW
    station_energy_price: Dict[int, float] = field(default_factory=dict) # $/kWh
    station_wait_time: Dict[int, float] = field(default_factory=dict)    # hours per visit
    station_wait_cost: Dict[int, float] = field(default_factory=dict)    # $ per visit (optional)
    station_detour_km: Dict[int, float] = field(default_factory=dict)    # extra km on arrival
def _euc2d(a, b) -> float:
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def build_distance_matrix(coords: List[Tuple[float,float]]) -> List[List[float]]:
    n = len(coords) - 1
    D = [[0.0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                D[i][j] = _euc2d(coords[i], coords[j])
    return D

def load_evrp(path: str) -> Problem:
    name = ""; vehicles = capacity = dimension = stations_cnt = None
    energy_capacity = energy_consumption = None
    coords: List[Tuple[float,float]] = [(-1.0, -1.0)]
    reading_coords = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if reading_coords:
                if line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 3:
                        idx = int(parts[0]); x = float(parts[1]); y = float(parts[2])
                        while len(coords) <= idx:
                            coords.append((0.0, 0.0))
                        coords[idx] = (x, y)
                else:
                    reading_coords = False
            elif line.startswith("NAME:"):
                name = line.split(":",1)[1].strip()
            elif line.startswith("VEHICLES:"):
                vehicles = int(line.split(":",1)[1].strip())
            elif line.startswith("DIMENSION:"):
                dimension = int(line.split(":",1)[1].strip())
            elif line.startswith("STATIONS:"):
                stations_cnt = int(line.split(":",1)[1].strip())
            elif line.startswith("CAPACITY:"):
                capacity = int(line.split(":",1)[1].strip())
            elif line.startswith("ENERGY_CAPACITY:"):
                energy_capacity = float(line.split(":",1)[1].strip())
            elif line.startswith("ENERGY_CONSUMPTION:"):
                energy_consumption = float(line.split(":",1)[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

    assert dimension and stations_cnt is not None
    assert len(coords) == dimension + 1, f"Expected {dimension} coords, got {len(coords)-1}"
    assert vehicles and capacity and energy_capacity is not None and energy_consumption is not None

    depot = 1
    num_customers = dimension - stations_cnt - 1
    customers = list(range(2, 2+num_customers))
    stations = list(range(dimension - stations_cnt + 1, dimension + 1))

    p = Problem(
        name=name, vehicles=vehicles, capacity=capacity,
        energy_capacity=energy_capacity, energy_consumption=energy_consumption,
        coords=coords, customers=customers, stations=stations, depot=depot
    )
    p.distance_matrix = build_distance_matrix(p.coords)
    return p

def apply_defaults(problem: Problem, *, charge_rate=None, energy_cost=None, waiting_cost=None, speed=None) -> Problem:
    problem.charge_rate  = (1.0 if problem.charge_rate  is None else problem.charge_rate)  if charge_rate  is None else charge_rate
    problem.energy_cost  = (0.0 if problem.energy_cost  is None else problem.energy_cost)  if energy_cost  is None else energy_cost
    default_w = 1.0 if problem.has_time_windows else 0.0
    problem.waiting_cost = (default_w if problem.waiting_cost is None else problem.waiting_cost) if waiting_cost is None else waiting_cost
    problem.speed        = (1.0 if problem.speed        is None else problem.speed)        if speed        is None else speed
    return problem
