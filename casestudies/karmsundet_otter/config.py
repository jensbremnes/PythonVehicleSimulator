"""config.py — all tunable parameters for the Karmsundet Otter case study.

Pure data module: no side effects, no imports beyond stdlib/pathlib.
"""
from pathlib import Path
import math

HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
BIF_PATH = (HERE / "../../lib/risk-aware-a-star/examples/karmsundet_usv/usv_risk.bif").resolve()
TABLE_PATH  = DATA_DIR / "table.npz"
CACHE_DIR   = DATA_DIR / "emodnet_cache"

# ---------------------------------------------------------------------------
# Study grid — mirrors run_example.py exactly (required for .npz compatibility)
# ---------------------------------------------------------------------------
WEST, SOUTH, EAST, NORTH = 5.00, 59.25, 5.55, 59.55
CRS        = "EPSG:4326"
RESOLUTION = 0.002   # degrees per pixel → ~150×275 grid

# ---------------------------------------------------------------------------
# Route (~2 km segment within Karmsundet strait)
# ---------------------------------------------------------------------------
START = (59.44, 5.19)   # Haugesund harbour
GOAL  = (59.29, 5.17)   # South exit

# ---------------------------------------------------------------------------
# Weather / environmental inputs (baseline / calm conditions)
# ---------------------------------------------------------------------------
WAVE_HEIGHT    = 0.5     # m   (calm baseline; increase to stress-test)
WIND_SPEED     = 4.0     # m/s
FOG_FRACTION   = 0.05
VESSEL_TRAFFIC = 2.0     # continuous medium traffic
CURRENT_SPEED  = 0.4     # m/s

# ---------------------------------------------------------------------------
# Planner — same settings as run_example.py for consistency
# ---------------------------------------------------------------------------
RISK_NODE        = "usv_risk"
RISK_STATE       = {"medium": 0.5, "high": 1.0}
RISK_WEIGHT      = 500.0
CONNECTIVITY     = 8
RISK_INFLATION_M = 200
RISK_EXPONENT    = 3
RISK_THRESHOLD   = 0.4

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SAMPLE_TIME      = 0.05    # s  (20 Hz; good Euler accuracy for otter dynamics)
MAX_SIM_TIME     = 12000.0  # s  safety cap
CAPTURE_RADIUS_M = 15.0    # m  waypoint switch distance
WP_STRIDE        = 1       # keep every Nth planner waypoint (1 = all)

# ---------------------------------------------------------------------------
# Otter vehicle
# ---------------------------------------------------------------------------
TAU_X        = 120.0  # N   surge force (moderate cruise)
V_CURRENT    = 0.0    # m/s environmental current
BETA_CURRENT = 0.0    # deg current direction

# ---------------------------------------------------------------------------
# Replan events — list of (trigger_time_s, {node_name: new_value, ...})
#
# Each entry fires once when simulation time >= trigger_time_s.
# node_name must match a BN node accepted by update_input().
# value is passed directly to geobn.ConstantSource(value).
# Empty list = no replanning.
# ---------------------------------------------------------------------------
REPLAN_EVENTS = [
    (300.0, {"wave_height": 2.5, "wind_speed": 15.0}),   # storm arrives at t=300 s
    # (600.0, {"vessel_traffic": 8.0}),                  # traffic spike example
]

# Human-readable label for each replan event (shown in plot legend).
# Must have the same length as REPLAN_EVENTS (or be empty).
REPLAN_LABELS = ["Storm arrival (t=300 s)"]
