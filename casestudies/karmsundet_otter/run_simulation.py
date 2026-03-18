"""run_simulation.py — Karmsundet Otter case study.

Plans a risk-optimal path through the Karmsundet strait using the risk-aware
A* planner backed by a Bayesian network, then simulates the Maritime Robotics
Otter USV following that path with LOS guidance.  Supports mid-simulation
replanning when environmental conditions change.

Run
---
    uv run python casestudies/karmsundet_otter/run_simulation.py

Outputs
-------
    Four matplotlib figures (shown interactively via plt.show()):
      Fig 1 — study area and risk overview (two-panel)
      Fig 2 — simulated trajectory on risk grid
      Fig 3 — time series (heading, speed, cross-track error)
      Fig 4 — animated playback of trajectory along planned route
"""
from __future__ import annotations

import math
import sys
from math import atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — find project root so we can import without installing
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_PROJECT_ROOT = (_HERE / "../..").resolve()
_SRC = _PROJECT_ROOT / "src"
_LIB_RAASTAR = _PROJECT_ROOT / "lib" / "risk-aware-a-star" / "src"

for _p in (_SRC, _LIB_RAASTAR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geobn
from risk_aware_a_star import RiskAwareAStarPlanner

from python_vehicle_simulator.vehicles.otter import otter
from python_vehicle_simulator.lib.gnc import attitudeEuler, ssa

import config


# ---------------------------------------------------------------------------
# Coordinate helpers (flat-earth, error < 0.1 m over 2 km at 59°N)
# ---------------------------------------------------------------------------

def latlon_to_ned(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert (lat, lon) to (north_m, east_m) relative to (lat0, lon0)."""
    north = (lat - lat0) * 111_320.0
    east  = (lon - lon0) * 111_320.0 * cos(radians(lat0))
    return north, east


def ned_to_latlon(north: float, east: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert (north_m, east_m) back to (lat, lon)."""
    lat = lat0 + north / 111_320.0
    lon = lon0 + east  / (111_320.0 * cos(radians(lat0)))
    return lat, lon


# ---------------------------------------------------------------------------
# Synthetic bathymetry fallback (mirrors run_example.py)
# ---------------------------------------------------------------------------

def _synthetic_depth(rows: int, cols: int) -> np.ndarray:
    depth = np.full((rows, cols), np.nan, dtype=np.float32)
    for r in range(rows):
        depth[r, 15:110] = 10.0 + 70.0 * (r / max(rows - 1, 1))
    return depth


# ---------------------------------------------------------------------------
# Planner setup
# ---------------------------------------------------------------------------

def setup_planner() -> tuple[RiskAwareAStarPlanner, object]:
    """Build the BN, fetch bathymetry, and return (planner, initial_result).

    Precomputes the inference table on first run; loads it on subsequent runs.
    Nodes listed in REPLAN_EVENTS are NOT frozen so update_input() works.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    H = round((config.NORTH - config.SOUTH) / config.RESOLUTION)
    W = round((config.EAST  - config.WEST)  / config.RESOLUTION)
    print(f"Karmsundet Otter case study  ({H}×{W} grid)")
    print(f"Start: {config.START}  →  Goal: {config.GOAL}")

    # ------------------------------------------------------------------
    # 1. Load BN and configure grid
    # ------------------------------------------------------------------
    bn = geobn.load(config.BIF_PATH)
    bn.set_grid(config.CRS, config.RESOLUTION,
                (config.WEST, config.SOUTH, config.EAST, config.NORTH))

    # ------------------------------------------------------------------
    # 2. EMODnet bathymetry (cached WCS fetch; synthetic fallback)
    # ------------------------------------------------------------------
    print("\nFetching EMODnet bathymetry ...")
    try:
        raw_depth = bn.fetch_raw(geobn.WCSSource(
            url="https://ows.emodnet-bathymetry.eu/wcs",
            layer="emodnet:mean",
            version="2.0.1",
            valid_range=(-1000.0, 100.0),
            cache_dir=config.CACHE_DIR,
        ))
        depth = -raw_depth
        depth[depth < 0] = np.nan
        print(f"  WCS ok  (depth range {np.nanmin(depth):.0f}–{np.nanmax(depth):.0f} m)")
    except Exception as exc:
        print(f"  WCS unavailable ({exc}); using synthetic channel depth.")
        depth = _synthetic_depth(H, W)

    sea_mask = np.isfinite(depth)
    print(f"  Sea pixels: {int(sea_mask.sum()):,} / {H * W:,}")

    # ------------------------------------------------------------------
    # 3. Register all BN inputs (order must match between precompute / load)
    # ------------------------------------------------------------------
    bn.set_input("water_depth",    geobn.ArraySource(depth))
    bn.set_input("vessel_traffic", geobn.ConstantSource(config.VESSEL_TRAFFIC))
    bn.set_input("current_speed",  geobn.ConstantSource(config.CURRENT_SPEED))
    bn.set_input("wave_height",    geobn.ConstantSource(config.WAVE_HEIGHT))
    bn.set_input("wind_speed",     geobn.ConstantSource(config.WIND_SPEED))
    bn.set_input("fog_fraction",   geobn.ConstantSource(config.FOG_FRACTION))

    bn.set_discretization("water_depth",    [0, 5, 20, 50, 200, 2000])
    bn.set_discretization("vessel_traffic", [0.0, 1.0, 3.0, 1000.0])
    bn.set_discretization("current_speed",  [0.0, 0.3, 1.0, 5.0])
    bn.set_discretization("wave_height",    [0.0, 0.5, 1.5, 15.0])
    bn.set_discretization("wind_speed",     [0.0, 5.0, 12.0, 50.0])
    bn.set_discretization("fog_fraction",   [0.0, 0.2, 0.6, 1.01])

    # ------------------------------------------------------------------
    # 4. Precomputed table: load if cached, otherwise compute and save
    # ------------------------------------------------------------------
    if config.TABLE_PATH.exists():
        print("\nLoading precomputed inference table ...")
        planner = RiskAwareAStarPlanner(
            bn,
            config.RISK_NODE,
            config.RISK_STATE,
            risk_weight=config.RISK_WEIGHT,
            connectivity=config.CONNECTIVITY,
            risk_inflation_m=config.RISK_INFLATION_M,
            risk_exponent=config.RISK_EXPONENT,
            risk_threshold=config.RISK_THRESHOLD,
        )
        planner.load_precomputed(config.TABLE_PATH)
    else:
        print("\nPrecomputing inference table (this takes ~30 s) ...")
        bn.precompute([config.RISK_NODE])
        bn.save_precomputed(config.TABLE_PATH)
        print(f"  Saved → {config.TABLE_PATH}")
        planner = RiskAwareAStarPlanner(
            bn,
            config.RISK_NODE,
            config.RISK_STATE,
            risk_weight=config.RISK_WEIGHT,
            connectivity=config.CONNECTIVITY,
            risk_inflation_m=config.RISK_INFLATION_M,
            risk_exponent=config.RISK_EXPONENT,
            risk_threshold=config.RISK_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # 5. Freeze static nodes (skip any that appear in REPLAN_EVENTS)
    # ------------------------------------------------------------------
    dynamic_nodes = {node for _, updates in config.REPLAN_EVENTS for node in updates}
    freezable = [n for n in ("water_depth", "vessel_traffic", "current_speed")
                 if n not in dynamic_nodes]
    if freezable:
        planner.freeze_static_nodes(*freezable)
        print(f"  Frozen static nodes: {freezable}")

    # ------------------------------------------------------------------
    # 6. Initial path plan
    # ------------------------------------------------------------------
    print(f"\nPlanning initial path {config.START} → {config.GOAL} ...")
    initial_result = planner.find_path(config.START, config.GOAL, return_coords="latlon")
    dist_km = initial_result.total_distance_px * config.RESOLUTION * 111.0
    print(f"  Waypoints: {len(initial_result.waypoints)}"
          f"  distance ≈ {dist_km:.2f} km"
          f"  cost = {initial_result.total_cost:.1f}")

    return planner, initial_result, sea_mask, depth


# ---------------------------------------------------------------------------
# LOS guidance
# ---------------------------------------------------------------------------

def los_guidance(
    eta: np.ndarray,
    waypoints_ned: list[tuple[float, float]],
    wp_idx: int,
) -> tuple[float | None, int]:
    """Line-of-sight guidance law.

    Returns (psi_d_rad, new_wp_idx).  Returns (None, wp_idx) when all
    waypoints have been captured.
    """
    pos_n, pos_e = eta[0], eta[1]

    # Advance past captured waypoints
    while wp_idx < len(waypoints_ned):
        wn, we = waypoints_ned[wp_idx]
        dist = math.hypot(wn - pos_n, we - pos_e)
        if dist <= config.CAPTURE_RADIUS_M:
            wp_idx += 1
        else:
            break

    if wp_idx >= len(waypoints_ned):
        return None, wp_idx

    wn, we = waypoints_ned[wp_idx]
    psi_d = atan2(we - pos_e, wn - pos_n)   # NED: 0=north, +π/2=east
    return psi_d, wp_idx


# ---------------------------------------------------------------------------
# Cross-track error
# ---------------------------------------------------------------------------

def compute_xte(
    eta: np.ndarray,
    waypoints_ned: list[tuple[float, float]],
    wp_idx: int,
) -> float:
    """Signed perpendicular distance from vehicle to current leg."""
    if wp_idx == 0 or wp_idx >= len(waypoints_ned):
        return 0.0

    # Leg from previous waypoint to current
    p1n, p1e = waypoints_ned[wp_idx - 1]
    p2n, p2e = waypoints_ned[wp_idx]

    leg_n = p2n - p1n
    leg_e = p2e - p1e
    leg_len = math.hypot(leg_n, leg_e)
    if leg_len < 1e-6:
        return 0.0

    # Vehicle offset from p1
    dn = eta[0] - p1n
    de = eta[1] - p1e

    # Signed cross-track error (2-D cross product / leg_len)
    xte = (leg_n * de - leg_e * dn) / leg_len
    return xte


# ---------------------------------------------------------------------------
# Waypoint downsampling
# ---------------------------------------------------------------------------

def _downsample(wps: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if config.WP_STRIDE <= 1:
        return wps
    sampled = wps[::config.WP_STRIDE]
    if sampled[-1] != wps[-1]:
        sampled.append(wps[-1])
    return sampled


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_otter_simulation(
    planner: RiskAwareAStarPlanner,
    initial_waypoints_ned: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, list, list, float, float]:
    """Simulate the Otter USV following a planned path with optional replanning.

    Returns
    -------
    simTime       : shape (N,)
    simData       : shape (N, 18)
    plan_segments : list of (start_step, waypoints_ned, result, label)
    replan_steps  : list of step indices where replanning fired
    lat0, lon0    : reference origin for NED conversion
    """
    # Reference origin = START (lat/lon)
    lat0, lon0 = config.START

    waypoints_ned = _downsample(initial_waypoints_ned)

    # Initial heading: point toward first waypoint
    if len(waypoints_ned) >= 2:
        wn, we = waypoints_ned[1]
        psi_init_rad = atan2(we - waypoints_ned[0][1], wn - waypoints_ned[0][0])
    else:
        psi_init_rad = 0.0
    psi_init_deg = degrees(psi_init_rad)

    # Otter vehicle
    vehicle = otter('headingAutopilot', psi_init_deg,
                    config.V_CURRENT, config.BETA_CURRENT, config.TAU_X)
    eta      = np.array([0, 0, 0, 0, 0, psi_init_rad], float)
    nu       = vehicle.nu.copy()
    u_actual = vehicle.u_actual.copy()

    N_max = int(config.MAX_SIM_TIME / config.SAMPLE_TIME) + 1

    # simData columns: [eta(6), nu(6), u_control(2), u_actual(2), psi_d(1), xte(1)] = 18
    simData = np.empty((N_max, 18), dtype=float)

    replan_events  = sorted(config.REPLAN_EVENTS, key=lambda x: x[0])
    replan_labels  = list(config.REPLAN_LABELS)
    plan_segments  = [(0, waypoints_ned, None, "Initial plan (calm)")]
    replan_steps   = []

    wp_idx = 0
    n_steps = 0

    print(f"\nSimulating Otter USV  (dt={config.SAMPLE_TIME} s, "
          f"max={config.MAX_SIM_TIME:.0f} s) ...")

    for step in range(N_max):
        t = step * config.SAMPLE_TIME

        # ------------------------------------------------------------------
        # Check replan triggers (consume each at most once)
        # ------------------------------------------------------------------
        while replan_events and t >= replan_events[0][0]:
            trigger_t, node_updates = replan_events.pop(0)
            label = replan_labels.pop(0) if replan_labels else f"Replan at t={trigger_t:.0f} s"
            print(f"  [t={t:.1f} s] Replan triggered: {node_updates}")

            for node, val in node_updates.items():
                planner.update_input(node, geobn.ConstantSource(val))

            cur_latlon = ned_to_latlon(eta[0], eta[1], lat0, lon0)
            try:
                new_result = planner.find_path(cur_latlon, config.GOAL,
                                               return_coords="latlon")
            except (ValueError, RuntimeError) as exc:
                print(f"    Replan failed ({exc}); continuing with existing path.")
                continue

            new_wps_ned = _downsample(
                [latlon_to_ned(lat, lon, lat0, lon0)
                 for lat, lon in new_result.waypoints]
            )
            waypoints_ned = new_wps_ned
            wp_idx = 0
            replan_steps.append(step)
            plan_segments.append((step, waypoints_ned, new_result, label))
            dist_km = new_result.total_distance_px * config.RESOLUTION * 111.0
            print(f"    New path: {len(waypoints_ned)} wps, "
                  f"dist≈{dist_km:.2f} km, cost={new_result.total_cost:.1f}")

        # ------------------------------------------------------------------
        # LOS guidance
        # ------------------------------------------------------------------
        psi_d, wp_idx = los_guidance(eta, waypoints_ned, wp_idx)
        if psi_d is None:
            print(f"  All waypoints captured at t={t:.1f} s  (step {step})")
            n_steps = step
            break

        vehicle.ref = degrees(psi_d)

        # ------------------------------------------------------------------
        # Control + dynamics
        # ------------------------------------------------------------------
        u_control = vehicle.headingAutopilot(eta, nu, config.SAMPLE_TIME)
        xte       = compute_xte(eta, waypoints_ned, wp_idx)

        # Store state
        simData[step, 0:6]   = eta
        simData[step, 6:12]  = nu
        simData[step, 12:14] = u_control
        simData[step, 14:16] = u_actual
        simData[step, 16]    = psi_d
        simData[step, 17]    = xte

        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, config.SAMPLE_TIME)
        eta = attitudeEuler(eta, nu, config.SAMPLE_TIME)

        n_steps = step + 1
    else:
        print(f"  Reached MAX_SIM_TIME ({config.MAX_SIM_TIME:.0f} s) without completing path.")

    simTime = np.arange(n_steps) * config.SAMPLE_TIME
    simData = simData[:n_steps]

    print(f"  Simulation complete: {n_steps} steps, {simTime[-1]:.1f} s")
    return simTime, simData, plan_segments, replan_steps, lat0, lon0


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_SEGMENT_COLORS = ["#0055cc", "#cc3300", "#008800", "#8800cc", "#cc8800"]


def _try_basemap(ax):
    try:
        import contextily
        contextily.add_basemap(
            ax,
            crs="EPSG:4326",
            source=contextily.providers.OpenTopoMap,
            zoom=12,
            attribution=False,
        )
    except Exception:
        ax.set_facecolor("#b8d4e8")


def plot_risk_overview(initial_result, sea_mask):
    """Figure 1 — study area (left) + risk grid and initial planned path (right)."""
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    extent = [config.WEST, config.EAST, config.SOUTH, config.NORTH]

    for ax in (ax_left, ax_right):
        ax.set_xlim(config.WEST, config.EAST)
        ax.set_ylim(config.SOUTH, config.NORTH)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        _try_basemap(ax)

    # Left panel: study area with start/goal markers
    ax_left.set_title("Study area", fontsize=11)
    ax_left.plot(config.START[1], config.START[0], "g*", markersize=14,
                 label="Start (Haugesund)", zorder=5)
    ax_left.plot(config.GOAL[1],  config.GOAL[0],  "rD", markersize=10,
                 label="Goal (south exit)", zorder=5)
    ax_left.legend(fontsize=8, loc="lower left")

    # Right panel: risk heatmap + planned path
    risk_display = np.where(sea_mask, initial_result.risk_grid, np.nan)
    im = ax_right.imshow(
        risk_display,
        cmap="RdYlGn_r",
        vmin=0.0, vmax=0.8,
        alpha=0.65,
        extent=extent,
        origin="upper",
        aspect="equal",
        zorder=3,
    )
    plt.colorbar(im, ax=ax_right, label="Risk probability", fraction=0.03, pad=0.02)
    ax_right.set_title("Risk grid and initial planned path", fontsize=11)

    wps = initial_result.waypoints
    if wps:
        lons = [w[1] for w in wps]
        lats = [w[0] for w in wps]
        ax_right.plot(lons, lats, "-", color=_SEGMENT_COLORS[0], linewidth=2.5,
                      alpha=0.9, zorder=6, label="Initial plan")
    ax_right.plot(config.START[1], config.START[0], "g*", markersize=14, zorder=7)
    ax_right.plot(config.GOAL[1],  config.GOAL[0],  "rD", markersize=10, zorder=7)
    ax_right.legend(fontsize=8, loc="lower left")

    fig.suptitle("Karmsundet — Otter USV route planning", fontsize=12)
    fig.tight_layout()
    return fig


def plot_trajectory(simData, plan_segments, replan_steps, sea_mask, lat0, lon0):
    """Figure 2 — simulated trajectory overlaid on risk grid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [config.WEST, config.EAST, config.SOUTH, config.NORTH]

    ax.set_xlim(config.WEST, config.EAST)
    ax.set_ylim(config.SOUTH, config.NORTH)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Karmsundet — Simulated Otter trajectory", fontsize=12)

    _try_basemap(ax)

    # Use final segment's risk grid (post-replan if replanning occurred)
    _, _, last_result, _ = plan_segments[-1]
    if last_result is not None:
        risk_grid = last_result.risk_grid
    else:
        _, _, first_result, _ = plan_segments[0]
        risk_grid = first_result.risk_grid if first_result is not None else None

    if risk_grid is not None:
        risk_display = np.where(sea_mask, risk_grid, np.nan)
        ax.imshow(
            risk_display,
            cmap="RdYlGn_r",
            vmin=0.0, vmax=0.8,
            alpha=0.65,
            extent=extent,
            origin="upper",
            aspect="equal",
            zorder=3,
        )

    # Planned path segments
    for seg_idx, (start_step, wps_ned, result, label) in enumerate(plan_segments):
        if result is not None:
            lons = [ned_to_latlon(n, e, lat0, lon0)[1] for n, e in wps_ned]
            lats = [ned_to_latlon(n, e, lat0, lon0)[0] for n, e in wps_ned]
        else:
            # Initial segment: use START + waypoints
            lons = [ned_to_latlon(n, e, lat0, lon0)[1] for n, e in wps_ned]
            lats = [ned_to_latlon(n, e, lat0, lon0)[0] for n, e in wps_ned]
        color = _SEGMENT_COLORS[seg_idx % len(_SEGMENT_COLORS)]
        ax.plot(lons, lats, "--", color=color, linewidth=2, alpha=0.7,
                zorder=5, label=label)

    # Simulated track
    n_steps = len(simData)
    sim_lats = np.zeros(n_steps)
    sim_lons = np.zeros(n_steps)
    for i in range(n_steps):
        sim_lats[i], sim_lons[i] = ned_to_latlon(
            simData[i, 0], simData[i, 1], lat0, lon0)
    ax.plot(sim_lons, sim_lats, "-", color="orange", linewidth=2.5,
            alpha=0.9, zorder=6, label="Simulated track")

    # Replan event markers on track
    for rs in replan_steps:
        if rs < n_steps:
            lat_r, lon_r = ned_to_latlon(simData[rs, 0], simData[rs, 1], lat0, lon0)
            ax.plot(lon_r, lat_r, "k^", markersize=10, zorder=8)

    # Start / goal / final position
    ax.plot(config.START[1], config.START[0], "g*", markersize=14, zorder=9, label="Start")
    ax.plot(config.GOAL[1],  config.GOAL[0],  "rD", markersize=10, zorder=9, label="Goal")
    if n_steps > 0:
        final_lat, final_lon = ned_to_latlon(simData[-1, 0], simData[-1, 1], lat0, lon0)
        ax.plot(final_lon, final_lat, "bs", markersize=9, zorder=9, label="Final position")

    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_timeseries(simTime, simData, replan_steps):
    """Figure 3 — time-series: heading, speed, cross-track error."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Karmsundet Otter — Time series", fontsize=12)

    t = simTime

    # Panel 1: heading
    ax = axes[0]
    psi_deg   = np.degrees([ssa(simData[i, 5]) for i in range(len(simData))])
    psi_d_deg = np.degrees(simData[:, 16])
    ax.plot(t, psi_deg,   label="ψ actual", color="#0055cc")
    ax.plot(t, psi_d_deg, "--", label="ψ_d (LOS)", color="#cc3300", alpha=0.8)
    ax.set_ylabel("Heading (deg)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: surge speed
    ax = axes[1]
    ax.plot(t, simData[:, 6], color="#008800", label="Surge u (m/s)")
    ax.set_ylabel("Surge speed (m/s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: cross-track error
    ax = axes[2]
    ax.plot(t, simData[:, 17], color="#8800cc", label="XTE (m)")
    ax.axhline(+config.CAPTURE_RADIUS_M, color="grey", linestyle="--",
               alpha=0.6, label=f"±{config.CAPTURE_RADIUS_M:.0f} m")
    ax.axhline(-config.CAPTURE_RADIUS_M, color="grey", linestyle="--", alpha=0.6)
    ax.set_ylabel("Cross-track error (m)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Replan vertical lines on all panels
    for rs_idx, rs in enumerate(replan_steps):
        if rs < len(simTime):
            t_r = simTime[rs]
            label_r = (config.REPLAN_LABELS[rs_idx]
                       if rs_idx < len(config.REPLAN_LABELS)
                       else f"Replan {rs_idx + 1}")
            for ax in axes:
                ax.axvline(t_r, color="red", linestyle=":", alpha=0.7)
            axes[0].text(t_r + 2, axes[0].get_ylim()[1] * 0.95,
                         label_r, color="red", fontsize=7, va="top")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

#: Simulation steps advanced per animation frame (30 × 0.05 s = 1.5 sim-s/frame).
_ANIM_STRIDE = 3000
#: Displayed frames per second.
_ANIM_FPS = 20


def animate_trajectory(
    simData: np.ndarray,
    plan_segments: list,
    sea_mask: np.ndarray,
    initial_result,
    lat0: float,
    lon0: float,
):
    """Figure 4 — animated playback of the Otter trajectory along the planned route.

    The basemap + risk heatmap + all planned route segments are drawn once as
    a static background.  Each frame advances the orange track line and moves
    the vessel marker forward in time.  The active planned-route segment (the
    one currently being followed) is highlighted; earlier segments are faded.

    Returns
    -------
    fig  : matplotlib Figure
    anim : FuncAnimation  (keep a reference alive until plt.show())
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    n_steps = len(simData)
    frame_indices = list(range(0, n_steps, _ANIM_STRIDE))
    if not frame_indices or frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    # Pre-compute full lat/lon track to avoid per-frame conversion
    sim_lats = np.empty(n_steps)
    sim_lons = np.empty(n_steps)
    for i in range(n_steps):
        sim_lats[i], sim_lons[i] = ned_to_latlon(
            simData[i, 0], simData[i, 1], lat0, lon0)

    # Heading at each step (radians) for the vessel direction indicator
    sim_psi = simData[:, 5]  # yaw (rad), NED convention

    fig, ax = plt.subplots(figsize=(9, 7))
    extent = [config.WEST, config.EAST, config.SOUTH, config.NORTH]

    ax.set_xlim(config.WEST, config.EAST)
    ax.set_ylim(config.SOUTH, config.NORTH)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Karmsundet — Otter USV trajectory (animation)", fontsize=12)

    _try_basemap(ax)

    # Static: risk heatmap
    risk_display = np.where(sea_mask, initial_result.risk_grid, np.nan)
    ax.imshow(
        risk_display,
        cmap="RdYlGn_r",
        vmin=0.0, vmax=0.8,
        alpha=0.55,
        extent=extent,
        origin="upper",
        aspect="equal",
        zorder=3,
    )

    # Static: start / goal markers
    ax.plot(config.START[1], config.START[0], "g*", markersize=14, zorder=9, label="Start")
    ax.plot(config.GOAL[1],  config.GOAL[0],  "rD", markersize=10, zorder=9, label="Goal")

    # Build one line artist per planned segment (updated each frame)
    # Each entry: (start_step_in_sim, lons_list, lats_list, line_artist)
    seg_artists: list[tuple[int, list, list, object]] = []
    for seg_idx, (start_step, wps_ned, _result, label) in enumerate(plan_segments):
        lons = [ned_to_latlon(n, e, lat0, lon0)[1] for n, e in wps_ned]
        lats = [ned_to_latlon(n, e, lat0, lon0)[0] for n, e in wps_ned]
        color = _SEGMENT_COLORS[seg_idx % len(_SEGMENT_COLORS)]
        line, = ax.plot([], [], "--", color=color, linewidth=2.0, alpha=0.4,
                        zorder=5, label=label)
        seg_artists.append((start_step, lons, lats, line))

    # Dynamic: growing orange track
    track_line, = ax.plot([], [], "-", color="orange", linewidth=2.5,
                          alpha=0.9, zorder=6, label="Otter track")

    # Dynamic: vessel position (filled circle) + heading arrow (quiver)
    vessel_dot, = ax.plot([], [], "o", color="darkorange", markersize=9, zorder=10)
    # quiver for heading — one arrow, updated via set_UVC / set_offsets
    hdg_quiver = ax.quiver(
        [], [], [], [],
        color="black", scale=0.3, scale_units="xy", width=0.004,
        zorder=11, headlength=4, headaxislength=3,
    )

    # Dynamic: time stamp
    time_text = ax.text(
        0.02, 0.97, "",
        transform=ax.transAxes,
        fontsize=9, va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )

    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()

    # Map sim step → index of the active planned segment
    seg_start_steps = [s for s, _, _, _ in seg_artists]

    def _active_seg(step: int) -> int:
        active = 0
        for i, ss in enumerate(seg_start_steps):
            if step >= ss:
                active = i
        return active

    def init():
        track_line.set_data([], [])
        vessel_dot.set_data([], [])
        time_text.set_text("")
        for _, lons, lats, line in seg_artists:
            line.set_data(lons, lats)
            line.set_alpha(0.4)
        return []

    def update(frame_i: int):
        step = frame_indices[frame_i]
        t = step * config.SAMPLE_TIME

        # Grow track
        track_line.set_data(sim_lons[: step + 1], sim_lats[: step + 1])

        # Vessel position
        lon_v, lat_v = sim_lons[step], sim_lats[step]
        vessel_dot.set_data([lon_v], [lat_v])

        # Heading arrow — convert NED psi (0=north, +east) to lon/lat delta
        psi = sim_psi[step]
        arrow_len = 0.008   # ~0.5 km in degrees at 59°N
        dlon = arrow_len * math.sin(psi)
        dlat = arrow_len * math.cos(psi)
        hdg_quiver.set_offsets([[lon_v, lat_v]])
        hdg_quiver.set_UVC([dlon], [dlat])

        # Time stamp
        time_text.set_text(f"t = {t:.0f} s")

        # Highlight active segment, fade others
        active = _active_seg(step)
        for seg_i, (_, lons, lats, line) in enumerate(seg_artists):
            if seg_i == active:
                line.set_alpha(0.85)
                line.set_linewidth(2.5)
            else:
                line.set_alpha(0.25)
                line.set_linewidth(1.5)

        return []

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 // _ANIM_FPS,
        blit=False,
        repeat=True,
    )
    return fig, anim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib.pyplot as plt

    # 1. Setup planner + initial path plan
    planner, initial_result, sea_mask, depth = setup_planner()

    # 2. Convert initial waypoints to NED
    lat0, lon0 = config.START
    initial_waypoints_ned = [
        latlon_to_ned(lat, lon, lat0, lon0)
        for lat, lon in initial_result.waypoints
    ]

    # 3. Run simulation
    simTime, simData, plan_segments, replan_steps, lat0, lon0 = run_otter_simulation(
        planner, initial_waypoints_ned
    )

    # 4. Fix up plan_segments: store initial_result in first segment
    if plan_segments and plan_segments[0][2] is None:
        start_step, wps, _, label = plan_segments[0]
        plan_segments[0] = (start_step, wps, initial_result, label)

    # 5. Plots
    print("\nGenerating figures ...")
    fig1 = plot_risk_overview(initial_result, sea_mask)
    fig2 = plot_trajectory(simData, plan_segments, replan_steps, sea_mask, lat0, lon0)
    fig3 = plot_timeseries(simTime, simData, replan_steps)
    fig4, anim4 = animate_trajectory(
        simData, plan_segments, sea_mask, initial_result, lat0, lon0)
    print(f"  Animation: {len(simData) // _ANIM_STRIDE + 1} frames "
          f"at {_ANIM_FPS} fps ({_ANIM_STRIDE * config.SAMPLE_TIME:.1f} s/frame)")

    plt.show()


if __name__ == "__main__":
    main()
