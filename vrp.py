"""
Streamlit app: Dynamic Excel-free VRP solver that accepts all inputs from the frontend and solves a capacity-constrained VRP
using OR-Tools.

Features implemented:
- User specifies number of locations and vehicles.
- User edits the distance matrix and weight (and optional volume) vectors directly in the UI using `st.data_editor`.
- Supports per-vehicle capacities and max distances.
- Allows node-drop penalty (to handle infeasibility).
- Uses OR-Tools to solve and displays a result table, colored route visualization, and CSV download.

Run: streamlit run vrp_no_excel_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="VRP — Interactive (No Excel)", layout="wide")
st.title("📦 Interactive VRP Solver — enter inputs on the frontend")

st.markdown("Enter the problem size and edit the distance matrix, weights and vehicle parameters. Then click Solve.")

# Problem size inputs
cols = st.columns(3)
with cols[0]:
    num_locations = st.number_input("Number of locations (including depot)", min_value=2, value=9, step=1)
with cols[1]:
    num_vehicles = st.number_input("Number of vehicles", min_value=1, value=3, step=1)
with cols[2]:
    depot_index = st.number_input("Depot index (0-based)", min_value=0, value=0, max_value=0 if num_locations<=1 else num_locations-1, step=1)

st.markdown("---")

# Create default distance matrix (symmetric) and default weights/volume
@st.cache_data
def default_matrix(n):
    # Create a symmetric matrix with zero diagonal and some distances
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
if 'distance_df' not in st.session_state or st.session_state.get('dm_n', None) != num_locations:
    st.session_state['distance_df'] = default_matrix(num_locations)
    st.session_state['dm_n'] = num_locations

distance_df = st.experimental_data_editor(st.session_state['distance_df'], key='distance_editor', num_rows="fixed")
# Ensure symmetry and integers
try:
    dist_mat = distance_df.astype(int).values
    # enforce symmetry and diagonal zero
    for i in range(len(dist_mat)):
        dist_mat[i,i] = 0
        for j in range(i+1, len(dist_mat)):
            # average to enforce symmetry if user entered asymmetric
            avg = int((dist_mat[i,j] + dist_mat[j,i]) / 2)
            dist_mat[i,j] = avg
            dist_mat[j,i] = avg
except Exception as e:
    st.error(f"Distance matrix must be numeric. {e}")
    st.stop()

st.markdown("---")

st.subheader("Package weights and optional volume")
if 'weights' not in st.session_state or st.session_state.get('w_n', None) != num_locations:
    st.session_state['weights'] = default_series(num_locations, 1, 50)
    st.session_state['w_n'] = num_locations
if 'volume' not in st.session_state or st.session_state.get('v_n', None) != num_locations:
    st.session_state['volume'] = default_series(num_locations, 1, 60)
    st.session_state['v_n'] = num_locations

weights_series = st.experimental_data_editor(pd.DataFrame({'Weight': st.session_state['weights']}), key='weights_editor', num_rows="fixed")
use_volume = st.checkbox("Use volume as second dimension", value=False)
if use_volume:
    volume_series = st.experimental_data_editor(pd.DataFrame({'Volume': st.session_state['volume']}), key='volume_editor', num_rows="fixed")
else:
    volume_series = None

# Vehicle parameters
st.markdown("---")
st.subheader("Vehicles — capacities and max distances")
# default vehicle capacities
if 'vehicle_caps' not in st.session_state or st.session_state.get('vc_n', None) != num_vehicles:
    st.session_state['vehicle_caps'] = [100]*num_vehicles
    st.session_state['vc_n'] = num_vehicles
if 'vehicle_maxdist' not in st.session_state or st.session_state.get('vd_n', None) != num_vehicles:
    st.session_state['vehicle_maxdist'] = [1000]*num_vehicles
    st.session_state['vd_n'] = num_vehicles

# allow user to edit per-vehicle
vehicle_caps = []
vehicle_maxdist = []
vehicle_cols = st.columns(min(4, num_vehicles))
for i in range(num_vehicles):
    with vehicle_cols[i % len(vehicle_cols)]:
        cap = st.number_input(f"Ve {i} capacity", value=int(st.session_state['vehicle_caps'][i]) if i < len(st.session_state['vehicle_caps']) else 100, min_value=1, step=1, key=f'cap_{i}')
        md = st.number_input(f"Ve {i} max dist", value=int(st.session_state['vehicle_maxdist'][i]) if i < len(st.session_state['vehicle_maxdist']) else 1000, min_value=1, step=1, key=f'md_{i}')
    vehicle_caps.append(int(cap))
    vehicle_maxdist.append(int(md))

st.markdown("---")

# Solver options
st.subheader("Solver options")
penalty = st.number_input("Penalty for dropping a node (0 means no drop allowed)", min_value=0, value=10000, step=100)
time_limit = st.slider("Time limit (seconds) for solver", min_value=1, max_value=120, value=15)
first_strategy = st.selectbox("First solution strategy", options=[
    'PATH_CHEAPEST_ARC', 'SAVINGS', 'SWEEP', 'CHRISTOFIDES', 'ALL_UNPERFORMED', 'BEST_INSERTION'
], index=0)
metaheuristic = st.selectbox("Local search metaheuristic", options=['GUIDED_LOCAL_SEARCH','TABU_SEARCH','SIMULATED_ANNEALING','AUTOMATIC'], index=0)

st.markdown("---")

# Solve button
if st.button("Solve VRP"):
    # Prepare data
    N = num_locations
    distances = dist_mat.tolist()
    weights = [int(x) for x in weights_series['Weight'].values]
    if use_volume:
        volumes = [int(x) for x in volume_series['Volume'].values]
    else:
        volumes = None

    # Basic validation
    if len(distances) != N or any(len(row)!=N for row in distances):
        st.error("Distance matrix dimensions do not match number of locations")
        st.stop()
    if len(weights) != N:
        st.error("Weights vector length must equal number of locations")
        st.stop()
    if volumes is not None and len(volumes) != N:
        st.error("Volume vector length must equal number of locations")
        st.stop()
    if depot_index < 0 or depot_index >= N:
        st.error("Depot index out of range")
        st.stop()

    # Build OR-Tools model
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distances[from_node][to_node])
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add weight capacity dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(weights[node])
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_caps, True, 'Weight')

    # Add volume if available
    if volumes is not None:
        def volume_callback(from_index):
            node = manager.IndexToNode(from_index)
            return int(volumes[node])
        vol_cb_idx = routing.RegisterUnaryTransitCallback(volume_callback)
        vol_caps = vehicle_caps[:]  # for simplicity allow same capacities or you could add separate inputs
        routing.AddDimensionWithVehicleCapacity(vol_cb_idx, 0, vol_caps, True, 'Volume')

    # Distance dimension and per-vehicle max
    routing.AddDimension(transit_callback_index, 0, max(vehicle_maxdist), True, 'Distance')
    distance_dimension = routing.GetDimensionOrDie('Distance')
    for i in range(num_vehicles):
        distance_dimension.CumulVar(routing.End(i)).SetMax(vehicle_maxdist[i])

    # Allow dropping nodes with penalty, except depot
    for node in range(N):
        if node == depot_index:
            continue
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # map string
    strategy_map = {
        'PATH_CHEAPEST_ARC': routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        'SAVINGS': routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        'SWEEP': routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        'CHRISTOFIDES': routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        'ALL_UNPERFORMED': routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        'BEST_INSERTION': routing_enums_pb2.FirstSolutionStrategy.BEST_INSERTION
    }
    search_parameters.first_solution_strategy = strategy_map.get(first_strategy, routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    meta_map = {
        'GUIDED_LOCAL_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        'TABU_SEARCH': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        'SIMULATED_ANNEALING': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        'AUTOMATIC': routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    }
    search_parameters.local_search_metaheuristic = meta_map.get(metaheuristic, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = int(time_limit)

    # Solve
    with st.spinner("Solving VRP..."):
        solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        st.error("No solution found. Try increasing time limit or relaxing constraints (reduce capacities or increase penalties).")
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
        # add final depot
        route_nodes.append(depot_index)
        # Only keep non-empty routes (more than just depot)
        if len(route_nodes) > 2:
            routes.append({'vehicle': v, 'route': route_nodes, 'distance': route_dist, 'load': route_load})
            total_distance += route_dist
            total_load += route_load

    st.subheader("Solution summary")
    if routes:
        df_routes = pd.DataFrame(routes)
        st.write(df_routes[['vehicle','route','distance','load']])
        st.write(f"Total distance: {total_distance} km  —  Total assigned load: {total_load} kg")

        # Build colored visualization
        st.subheader("Route visualization")
        # place nodes on circle
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        fig, ax = plt.subplots(figsize=(7,7))
        # draw nodes
        for i,(x,y) in enumerate(zip(xs,ys)):
            size = 60 if i==depot_index else 40
            ax.scatter(x,y, s=size, zorder=3)
            ax.text(x, y, f"{i}
W:{weights[i]}" + (f"
V:{volumes[i]}" if volumes is not None else ''), fontsize=9, ha='center', va='center')
        # colors for vehicles
        cmap = plt.get_cmap('tab10')
        for r in routes:
            veh = r['vehicle']
            route = r['route']
            coords = [(xs[n], ys[n]) for n in route]
            xs_r, ys_r = zip(*coords)
            ax.plot(xs_r, ys_r, linewidth=2.5, alpha=0.9, color=cmap(veh % 10), zorder=2)
            # mark start (depot->first)
            ax.scatter(xs_r[0], ys_r[0], s=120, marker='X', color=cmap(veh%10), zorder=4)
        ax.set_axis_off()
        st.pyplot(fig)

        # CSV download
        csv_buf = io.StringIO()
        df_routes.to_csv(csv_buf, index=False)
        st.download_button("Download routes CSV", csv_buf.getvalue(), file_name="vrp_routes.csv")

    else:
        st.info("No vehicles used (all nodes possibly dropped). Increase penalty or add capacity.")

    # Add short report (2-3 paragraphs) summarizing parameters and results
    st.markdown("---")
    st.subheader("Auto-generated short report")
    report = f"Problem: {N} locations (depot {depot_index}), {num_vehicles} vehicles. Total demand (excluding depot): {sum(weights)-weights[depot_index]} units.
"
    report += f"Vehicle capacities: {vehicle_caps}. Vehicle max distances: {vehicle_maxdist}.
"
    report += f"Solver settings: time_limit={time_limit}s, first_solution={first_strategy}, metaheuristic={metaheuristic}.
"
    report += f"Result: vehicles used {len(routes)}, total distance {total_distance} km, total assigned load {total_load} units."
    st.code(report)

st.markdown("---")
st.caption("Interactive VRP: enter data on the frontend — no Excel required. This app is ideal for demos and classroom projects.")
