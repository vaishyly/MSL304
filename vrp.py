"""
Streamlit app: Dynamic Excel-free VRP solver that accepts all inputs from the frontend and solves a capacity-constrained VRP
using OR-Tools.

Features implemented:
- User specifies number of locations and vehicles.
- User edits the distance matrix and weight (and optional volume) vectors directly in the UI using `st.data_editor`.
- Supports per-vehicle capacities and max distances.
- Allows node-drop penalty (to handle infeasibility).
- Uses OR-Tools to solve and displays a result table, colored route visualization, and CSV download.

Run locally:
    streamlit run vrp.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="VRP — Interactive (No Excel)", layout="wide")
st.title("📦 Interactive VRP Solver — Enter Inputs")

st.markdown("Enter the problem size and edit the distance matrix, weights, and vehicle parameters. Then click **Solve**.")

# Problem size inputs
cols = st.columns(3)
with cols[0]:
    num_locations = st.number_input("Number of locations (including depot)", min_value=2, value=9, step=1)
with cols[1]:
    num_vehicles = st.number_input("Number of vehicles", min_value=1, value=3, step=1)
with cols[2]:
    depot_index = st.number_input("Depot index (0-based)", min_value=0, value=0, max_value=num_locations - 1, step=1)

st.markdown("---")

# Create default distance matrix (symmetric)
@st.cache_data
def default_matrix(n):
    rng = np.random.default_rng(42)
    base = rng.integers(10, 150, size=(n, n))
    mat = (base + base.T) // 2
    np.fill_diagonal(mat, 0)
    return pd.DataFrame(mat, index=[str(i) for i in range(n)], columns=[str(i) for i in range(n)])


@st.cache_data
def default_series(n, low=1, high=50):
    rng = np.random.default_rng(7)
    return pd.Series(rng.integers(low, high, size=n), index=[str(i) for i in range(n)])


# Initialize editor widgets
st.subheader("Edit Distance Matrix")
if "distance_df" not in st.session_state or st.session_state.get("dm_n", None) != num_locations:
    st.session_state["distance_df"] = default_matrix(num_locations)
    st.session_state["dm_n"] = num_locations

distance_df = st.data_editor(st.session_state["distance_df"], key="distance_editor", num_rows="fixed")

try:
    dist_mat = distance_df.astype(int).values
    for i in range(len(dist_mat)):
        dist_mat[i, i] = 0
        for j in range(i + 1, len(dist_mat)):
            avg = int((dist_mat[i, j] + dist_mat[j, i]) / 2)
            dist_mat[i, j] = avg
            dist_mat[j, i] = avg
except Exception as e:
    st.error(f"Distance matrix must be numeric. {e}")
    st.stop()

st.markdown("---")

st.subheader("Package weights and optional volume")
if "weights" not in st.session_state or st.session_state.get("w_n", None) != num_locations:
    st.session_state["weights"] = default_series(num_locations, 1, 50)
    st.session_state["w_n"] = num_locations
if "volume" not in st.session_state or st.session_state.get("v_n", None) != num_locations:
    st.session_state["volume"] = default_series(num_locations, 1, 60)
    st.session_state["v_n"] = num_locations

weights_series = st.data_editor(
    pd.DataFrame({"Weight": st.session_state["weights"]}), key="weights_editor", num_rows="fixed"
)
use_volume = st.checkbox("Use volume as second dimension", value=False)
if use_volume:
    volume_series = st.data_editor(
        pd.DataFrame({"Volume": st.session_state["volume"]}), key="volume_editor", num_rows="fixed"
    )
else:
    volume_series = None

st.markdown("---")

# Vehicle parameters
st.subheader("Vehicles — capacities and max distances")
if "vehicle_caps" not in st.session_state or st.session_state.get("vc_n", None) != num_vehicles:
    st.session_state["vehicle_caps"] = [100] * num_vehicles
    st.session_state["vc_n"] = num_vehicles
if "vehicle_maxdist" not in st.session_state or st.session_state.get("vd_n", None) != num_vehicles:
    st.session_state["vehicle_maxdist"] = [1000] * num_vehicles
    st.session_state["vd_n"] = num_vehicles

vehicle_caps = []
vehicle_maxdist = []
vehicle_cols = st.columns(min(4, num_vehicles))
for i in range(num_vehicles):
    with vehicle_cols[i % len(vehicle_cols)]:
        cap = st.number_input(f"Vehicle {i} capacity", value=int(st.session_state["vehicle_caps"][i]), min_value=1, step=1)
        md = st.number_input(f"Vehicle {i} max distance", value=int(st.session_state["vehicle_maxdist"][i]), min_value=1, step=1)
    vehicle_caps.append(int(cap))
    vehicle_maxdist.append(int(md))

st.markdown("---")

# Solver options
st.subheader("Solver options")
penalty = st.number_input("Penalty for dropping a node (0 means no drop allowed)", min_value=0, value=10000, step=100)
time_limit = st.slider("Time limit (seconds) for solver", min_value=1, max_value=120, value=15)
first_strategy = st.selectbox(
    "First solution strategy",
    ["PATH_CHEAPEST_ARC", "SAVINGS", "SWEEP", "CHRISTOFIDES", "ALL_UNPERFORMED", "BEST_INSERTION"],
)
metaheuristic = st.selectbox(
    "Local search metaheuristic",
    ["GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING", "AUTOMATIC"],
)

st.markdown("---")

# Solve VRP
if st.button("Solve VRP"):
    N = num_locations
    distances = dist_mat.tolist()
    weights = [int(x) for x in weights_series["Weight"].values]
    volumes = [int(x) for x in volume_series["Volume"].values] if use_volume else None

    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return int(distances[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Weight dimension
    def demand_callback(from_index):
        return int(weights[manager.IndexToNode(from_index)])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_caps, True, "Weight")

    # Volume dimension (optional)
    if volumes is not None:
        def volume_callback(from_index):
            return int(volumes[manager.IndexToNode(from_index)])
        vol_cb_idx = routing.RegisterUnaryTransitCallback(volume_callback)
        routing.AddDimensionWithVehicleCapacity(vol_cb_idx, 0, vehicle_caps, True, "Volume")

    # Distance constraints
    routing.AddDimension(transit_callback_index, 0, max(vehicle_maxdist), True, "Distance")
    distance_dimension = routing.GetDimensionOrDie("Distance")
    for i in range(num_vehicles):
        distance_dimension.CumulVar(routing.End(i)).SetMax(vehicle_maxdist[i])

    # Dropping nodes
    for node in range(N):
        if node != depot_index:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Solver parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    strategy_map = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "SWEEP": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "ALL_UNPERFORMED": routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        "BEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION,
    }
    search_parameters.first_solution_strategy = strategy_map[first_strategy]

    meta_map = {
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    }
    search_parameters.local_search_metaheuristic = meta_map[metaheuristic]
    search_parameters.time_limit.seconds = int(time_limit)

    with st.spinner("Solving VRP..."):
        solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        st.error("No solution found. Try increasing time limit or relaxing constraints.")
        st.stop()

    # Extract routes
    routes = []
    total_distance = 0
    total_load = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route_nodes = []
        route_load = 0
        route_dist = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            route_load += weights[node]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                curr_node = manager.IndexToNode(previous_index)
                next_node = manager.IndexToNode(index)
                route_dist += distances[curr_node][next_node]
        route_nodes.append(depot_index)
        if len(route_nodes) > 2:
            routes.append({"vehicle": v, "route": route_nodes, "distance": route_dist, "load": route_load})
            total_distance += route_dist
            total_load += route_load

    st.subheader("Solution Summary")
    if routes:
        df_routes = pd.DataFrame(routes)
        st.write(df_routes[["vehicle", "route", "distance", "load"]])
        st.success(f"Total Distance: {total_distance} km | Total Load: {total_load}")

        # Visualization
        st.subheader("Route Visualization")
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)
        fig, ax = plt.subplots(figsize=(7, 7))
        cmap = plt.get_cmap("tab10")
        for i, (x, y) in enumerate(zip(xs, ys)):
            size = 80 if i == depot_index else 50
            ax.scatter(x, y, s=size, color="black" if i == depot_index else "gray", zorder=3)
            label = f"{i}\\nW:{weights[i]}" + (f"\\nV:{volumes[i]}" if volumes is not None else "")
            ax.text(x, y, label, fontsize=8, ha="center", va="center")

        for r in routes:
            veh = r["vehicle"]
            route = r["route"]
            coords = [(xs[n], ys[n]) for n in route]
            xr, yr = zip(*coords)
            ax.plot(xr, yr, linewidth=2.5, color=cmap(veh % 10), label=f"Vehicle {veh}")
        ax.legend()
        ax.axis("off")
        st.pyplot(fig)

        # CSV download
        csv_buf = io.StringIO()
        df_routes.to_csv(csv_buf, index=False)
        st.download_button("Download Routes CSV", csv_buf.getvalue(), file_name="vrp_routes.csv")

        # Short report
        st.subheader("Auto-generated Report Summary")
        report = f"""
        Problem: {N} locations (depot {depot_index}), {num_vehicles} vehicles.
        Vehicle capacities: {vehicle_caps}
        Vehicle max distances: {vehicle_maxdist}
        Solver: time_limit={time_limit}s, first_solution={first_strategy}, metaheuristic={metaheuristic}
        Result: {len(routes)} vehicles used, total distance {total_distance} km, total load {total_load}.
        """
        st.code(report)

    else:
        st.warning("No routes found! Try changing parameters or constraints.")

st.markdown("---")
st.caption("Developed for MSL304 — Interactive Vehicle Routing Problem Solver (Streamlit + OR-Tools).")
