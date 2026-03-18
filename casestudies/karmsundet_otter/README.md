# Karmsundet Otter Case Study

End-to-end demonstration of risk-aware A\* path planning combined with Otter USV
simulation in the **Karmsundet strait**, near Haugesund, Norway.

---

## Overview

The case study chains two components:

1. **Risk-aware A\* planner** (`risk-aware-a-star` library) — queries a Bayesian
   network backed by real EMODnet bathymetry to find a route that avoids shallow
   water, high-traffic corridors, and bad weather.
2. **Otter USV dynamics** (`python_vehicle_simulator`) — simulates the 2 m
   Maritime Robotics Otter following the planned waypoints with a PID heading
   autopilot and LOS guidance.

Karmsundet was chosen because it is already in the literature for USV/ship
risk-based navigation and the BIF, discretisations, and EMODnet setup are
already validated in `lib/risk-aware-a-star/examples/karmsundet_usv/`.

---

## Quick start

```bash
uv pip install -e ".[casestudies]"
uv run python casestudies/karmsundet_otter/run_simulation.py
```

**First run** precomputes the Bayesian network inference table (~30 s) and
saves it to `data/table.npz`. **Subsequent runs** load the table instantly.

Four matplotlib figures open when the simulation completes.

---

## What you see

| Figure | Content |
|--------|---------|
| **Figure 1** | Two-panel plot: left panel shows the risk heatmap (RdYlGn_r) with OpenTopoMap topographic basemap and the initial planned path; right panel shows the same route on the raw risk grid. Green star = START, red diamond = GOAL. |
| **Figure 2** | Same heatmap (final/post-replan risk grid) with all planned path segments (dashed, each in a different colour) and the simulated Otter track (orange). Black triangles mark replan events. |
| **Figure 3** | Time series: actual heading vs LOS setpoint (top), surge speed (middle), cross-track error ± capture radius (bottom). Vertical red dashed lines mark replan events. |
| **Figure 4** | Animated playback of the Otter's trajectory over the topographic basemap and risk heatmap. The orange track grows frame by frame; the vessel marker moves with a heading arrow. The active planned-route segment is highlighted while past segments are faded. Loops continuously. |

---

## Changing the route

Edit `START` and `GOAL` in `config.py` (WGS84 lat/lon tuples):

```python
START = (59.44, 5.19)   # Haugesund harbour
GOAL  = (59.29, 5.17)   # South exit
```

Both points must lie within the study grid
`WEST=5.00, SOUTH=59.25, EAST=5.55, NORTH=59.55`.

---

## Changing weather / risk conditions

Edit the constants in the `# Weather` block of `config.py`:

```python
WAVE_HEIGHT    = 0.5     # m
WIND_SPEED     = 4.0     # m/s
FOG_FRACTION   = 0.05
VESSEL_TRAFFIC = 2.0     # continuous medium traffic
CURRENT_SPEED  = 0.4     # m/s
```

These set the **baseline** conditions used for the initial plan.

---

## Replanning

`REPLAN_EVENTS` in `config.py` is a list of `(trigger_time_s, {node: value})`
tuples.  When the simulation time reaches `trigger_time_s` the script:

1. Calls `planner.update_input(node, ConstantSource(value))` for each changed node.
2. Re-runs `find_path()` from the Otter's **current position** to the original GOAL.
3. Replaces the active waypoint list and resets the waypoint index.
4. Records the event for visualisation (coloured path segments + dashed lines).

### Example events

```python
# Storm arrives at t = 300 s
REPLAN_EVENTS = [
    (300.0, {"wave_height": 2.5, "wind_speed": 15.0}),
]
REPLAN_LABELS = ["Storm arrival (t=300 s)"]

# Storm + later traffic spike
REPLAN_EVENTS = [
    (300.0, {"wave_height": 2.5, "wind_speed": 15.0}),
    (600.0, {"vessel_traffic": 8.0}),
]
REPLAN_LABELS = ["Storm", "Traffic spike"]
```

Any BN node accepted by `update_input()` can appear in `REPLAN_EVENTS`.
Nodes listed there are automatically **not frozen** so `update_input()` works
correctly — no other changes are needed.

Set `REPLAN_EVENTS = []` to disable replanning entirely.

---

## Planner parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RISK_WEIGHT` | 500.0 | Higher = more risk-averse (longer but safer routes) |
| `RISK_INFLATION_M` | 200 | Radius (m) around high-risk cells that are also penalised |
| `RISK_THRESHOLD` | 0.4 | Cells with risk > threshold are treated as impassable |
| `CONNECTIVITY` | 8 | 4 = cardinal only, 8 = cardinal + diagonal moves |

---

## First-run vs subsequent runs

On first run the script precomputes all `1,215` BN inference combinations and
saves the result to `data/table.npz`.  This takes roughly 30 s on a modern
laptop.  Subsequent runs load the table in < 1 s.

Delete `data/table.npz` to force recomputation (e.g. after changing the BIF or
discretisations).

---

## Extending to other locations

To adapt the case study to a different waterway:

1. **BIF**: replace `BIF_PATH` in `config.py` with your own `.bif` file.
2. **Grid bounds**: update `WEST, SOUTH, EAST, NORTH` and `RESOLUTION`.
3. **Route**: update `START` and `GOAL`.
4. **BN inputs**: update the `bn.set_input()` and `bn.set_discretization()` calls
   in `setup_planner()` to match your BN's node names and value ranges.
5. Delete `data/table.npz` so the table is recomputed for the new setup.
