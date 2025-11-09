"""
OptiRoute - Advanced Streamlit VRP App (Decomposed + Stochastic + KPI Dashboard)

Features:
- Random demo customers / editable tables + CSV upload
- Stage 1: Customer -> Vehicle assignment using KMeans
- Stage 2: Per-cluster routing using OR-Tools
- Stochastic sensitivity: ±% variation in demands and travel times
- KPI Dashboard: cost, average lateness, utilization, sensitivity charts
- Interactive Plotly visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------------------------
# Helpful: check imports & instructions if something missing
# ---------------------------
REQUIRED_PKGS = ["streamlit", "pandas", "numpy", "scikit-learn", "ortools", "plotly"]
# (We don't programmatically install to respect environment policies; user should pip install requirements.)

# ---------------------------
# Utility functions: VRP solver (per-cluster), stochastic variation, KPIs
# ---------------------------

def solve_vrp_ortools(dist_matrix, demands, vehicle_capacity, penalty=10000, time_limit=10):
    """
    Solve a capacitated VRP for a single vehicle cluster including the depot.
    Input:
      - dist_matrix: square numpy array (n x n) with distances (integers)
      - demands: list/array of length n (demand at each node). index 0 is depot (demand usually 0)
      - vehicle_capacity: scalar capacity for the single vehicle for this cluster (we will treat it as a single vehicle VRP)
      - penalty: disjunction penalty for dropping nodes (large => force serve)
      - time_limit: seconds solver will run
    Returns:
      - routes: list of routes (each route is list of node indices in local indexing)
      - metrics: dict {total_distance, served_nodes_count, total_load}
    Note:
      This function builds a small VRP with as many vehicles as needed (we will create 1 vehicle per cluster).
    """
    n = len(dist_matrix)
    if n == 0:
        return [], {"total_distance": 0, "served": 0, "total_load": 0}

    # If only depot exists, nothing to do
    if n == 1:
        return [[0, 0]], {"total_distance": 0, "served": 0, "total_load": 0}

    # Build manager for single vehicle
    num_vehicles = 1
    depot_idx = 0
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])

    transit_callback_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_idx)

    # Demand callback
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return int(demands[node])

    demand_callback_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_idx, 0, [vehicle_capacity], True, "Capacity")

    # Allow dropping nodes (except depot)
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = int(time_limit)

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        # no solution found
        return [], {"total_distance": None, "served": 0, "total_load": 0}

    # extract route(s)
    routes = []
    total_distance = 0
    total_load = 0
    served = 0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        route_load = 0
        route_dist = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            route_load += demands[node]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                curr_node = manager.IndexToNode(previous_index)
                next_node = manager.IndexToNode(index)
                route_dist += dist_matrix[curr_node][next_node]
        route.append(depot_idx)
        # count served nodes (exclude depot at start and end)
        served_local = sum(1 for nd in route if nd != depot_idx) - (1 if len(route) > 1 and route[0] == depot_idx else 0)
        if len(route) > 2:
            routes.append(route)
            total_distance += route_dist
            total_load += route_load
            served += served_local

    metrics = {"total_distance": total_distance, "served": served, "total_load": total_load}
    return routes, metrics

def apply_variation_matrix(base_matrix, var_pct, rng):
    """
    Apply multiplicative uniform variation of +/- var_pct to each positive entry of base_matrix.
    base_matrix: numpy array
    var_pct: e.g., 0.1 for +/-10%
    rng: numpy random generator
    """
    if var_pct <= 0:
        return base_matrix.copy()
    # generate factor matrix in [1-var_pct, 1+var_pct]
    noise = rng.uniform(1 - var_pct, 1 + var_pct, size=base_matrix.shape)
    varied = np.round(base_matrix * noise).astype(int)
    # Ensure zeros stay zero
    varied[base_matrix == 0] = 0
    # zero diagonal
    np.fill_diagonal(varied, 0)
    return varied

def compute_kpis(routes_all, cluster_assignments, cluster_demands, cluster_caps, cluster_distances):
    """
    Compute simple KPIs:
      - total_cost (distance)
      - avg_delay (we do not model explicit time windows here, so avg_delay = 0)
      - utilization % per vehicle = total_load / capacity
    For this simplified KPI: lateness is not modelled; we keep lateness = 0 but structure for future.
    """
    total_distance = 0
    total_load = 0
    utilizations = []
    served_count = 0
    for vidx, rinfo in enumerate(routes_all):
        # rinfo: dictionary with 'routes' list for that vehicle, 'metrics'
        if rinfo is None or 'metrics' not in rinfo:
            utilizations.append(0.0)
            continue
        metrics = rinfo['metrics']
        td = metrics.get('total_distance', 0) or 0
        tload = metrics.get('total_load', 0) or 0
        total_distance += td
        total_load += tload
        cap = cluster_caps[vidx] if vidx < len(cluster_caps) else cluster_caps[-1]
        utilization = 100.0 * tload / cap if cap > 0 else 0.0
        utilizations.append(utilization)
        served_count += metrics.get('served', 0) or 0

    avg_util = np.mean(utilizations) if len(utilizations) > 0 else 0
    kpis = {
        "total_distance": total_distance,
        "total_load": total_load,
        "vehicles_used": sum(1 for u in utilizations if u > 0),
        "avg_utilization_pct": avg_util,
        "served_count": served_count
    }
    return kpis

# ---------------------------
# Streamlit app UI & flow
# ---------------------------
st.set_page_config(page_title="OptiRoute - Decomposed VRP + Stochastic KPIs", layout="wide")
st.title("OptiRoute — Decomposed VRP (Clustering + Routing) with Stochastic Sensitivity & KPI Dashboard")
st.markdown("Team: Vaishali Anand, Vipul Yadav, Kundan, Arpit Agrawal — IIT Delhi, MSL304")

# Sidebar: basic controls
st.sidebar.header("Demo / Data options")
demo_customers = st.sidebar.number_input("Demo: number of customer nodes (excluding depot)", min_value=2, max_value=50, value=8, step=1)
demo_vehicles = st.sidebar.number_input("Demo: number of vehicles", min_value=1, max_value=6, value=3, step=1)
seed = st.sidebar.number_input("Random seed (demo)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.write("You can upload a CSV with columns: id,x,y,demand  (id should include depot as 0).")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# Main: generate or load data
if 'base_df' not in st.session_state or st.session_state.get("last_demo_params", None) != (demo_customers, demo_vehicles, seed):
    # create demo dataset
    rng = np.random.default_rng(seed)
    # create depot at center and customers randomized
    depot = {"id": 0, "x": 50.0, "y": 50.0, "demand": 0}
    customers = []
    for i in range(1, demo_customers + 1):
        # place customers around the depot
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(5, 40)
        x = depot['x'] + r * np.cos(angle)
        y = depot['y'] + r * np.sin(angle)
        demand = int(rng.integers(2, 30))
        customers.append({"id": i, "x": float(x), "y": float(y), "demand": int(demand)})
    df = pd.DataFrame([depot] + customers)
    st.session_state['base_df'] = df
    st.session_state['last_demo_params'] = (demo_customers, demo_vehicles, seed)

# If user uploaded CSV, let them replace the data
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        # Expecting columns: id,x,y,demand
        if not set(["id", "x", "y", "demand"]).issubset(set(uploaded_df.columns)):
            st.sidebar.error("CSV must contain columns: id, x, y, demand")
        else:
            st.session_state['base_df'] = uploaded_df.copy()
    except Exception as e:
        st.sidebar.error(f"Failed to parse CSV: {e}")

# show editable table
st.subheader("1) Locations & Demands (editable)")
df_nodes = st.data_editor(st.session_state['base_df'], num_rows="dynamic", key="nodes_editor")
# normalize column types
df_nodes = df_nodes.rename(columns={c: c.strip() for c in df_nodes.columns})
if 'id' not in df_nodes.columns:
    st.error("Data must include 'id' column. Edit to include id (0 is depot).")
    st.stop()

# ensure depot exists and is id 0; if not, try to set first row as depot
if 0 not in df_nodes['id'].values:
    st.warning("No depot with id 0 found. Treating first row as depot and assigning id 0.")
    df_nodes.loc[df_nodes.index[0], 'id'] = 0

# build coordinate matrix
nodes = df_nodes.sort_values('id').reset_index(drop=True)
coords = nodes[['x', 'y']].to_numpy()
demands = nodes['demand'].astype(int).to_numpy()
node_ids = nodes['id'].astype(int).to_list()

# compute distance matrix (Euclidean, rounded)
def euclid_matrix(coords):
    n = coords.shape[0]
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 0
            else:
                d = np.linalg.norm(coords[i] - coords[j])
                mat[i, j] = int(round(d))
    return mat

base_dist_matrix = euclid_matrix(coords)

# Vehicle parameters editable
st.subheader("2) Vehicle Parameters")
num_vehicles = st.number_input("Number of vehicles", min_value=1, value=demo_vehicles, step=1)
vehicle_capacity_default = st.number_input("Default vehicle capacity (applies to all unless edited)", min_value=1, value=100, step=1)
vehicle_caps = []
vehicle_maxdist = []
vehicle_cols = st.columns(min(4, num_vehicles))
for i in range(num_vehicles):
    with vehicle_cols[i % len(vehicle_cols)]:
        cap = st.number_input(f"Vehicle {i} capacity", value=vehicle_capacity_default, min_value=1, key=f'cap_{i}')
        md = st.number_input(f"Vehicle {i} max distance", value=1000, min_value=1, key=f'md_{i}')
    vehicle_caps.append(int(cap))
    vehicle_maxdist.append(int(md))

# Clustering options
st.subheader("3) Stage 1: Customer assignment (Clustering)")
cluster_method = st.selectbox("Clustering method", options=["KMeans (default)", "Greedy capacity-aware"], index=0)
do_cluster = st.button("Run clustering and assign customers to vehicles")

# Sensitivity / stochastic controls
st.subheader("4) Stochastic Sensitivity (What-if)")
col_a, col_b = st.columns(2)
with col_a:
    demand_var_pct = st.slider("Demand variation ±% (simulate)", min_value=0, max_value=50, value=10, step=1)
with col_b:
    travel_var_pct = st.slider("Travel time/distance variation ±% (simulate)", min_value=0, max_value=50, value=10, step=1)

st.markdown("Click 'Run full pipeline' to perform clustering → routing → KPIs under current stochastic settings.")
run_pipeline = st.button("Run full pipeline (clustering → routing → KPIs)")

# Keep data in session for reuse
st.session_state['base_df'] = nodes
st.session_state['coords'] = coords
st.session_state['demands'] = demands
st.session_state['dist_matrix'] = base_dist_matrix

# Helper: greedy capacity-aware assignment
def greedy_capacity_assign(coords, demands, num_vehicles, vehicle_caps):
    """
    Simple greedy assignment: sort customers by demand desc, assign to vehicle with most remaining capacity
    """
    n = coords.shape[0]
    # exclude depot if present at index 0
    node_indices = list(range(n))
    # assume depot at index 0
    customers = [i for i in node_indices if i != 0]
    remaining = {v: vehicle_caps[v] for v in range(num_vehicles)}
    assignment = {v: [] for v in range(num_vehicles)}
    # sort customers by demand descending
    customers_sorted = sorted(customers, key=lambda i: demands[i], reverse=True)
    for c in customers_sorted:
        # assign to vehicle with most remaining capacity that can take it
        best_v = max(range(num_vehicles), key=lambda v: remaining[v])
        assignment[best_v].append(c)
        remaining[best_v] -= demands[c]
    return assignment

# Run clustering (either on button or pipeline) and/or use cached cluster
if 'cluster_assignment' not in st.session_state:
    st.session_state['cluster_assignment'] = None

if do_cluster or run_pipeline:
    # perform clustering
    coords_local = st.session_state['coords']
    n_nodes = coords_local.shape[0]
    # cluster only on customers (exclude depot at index of id==0)
    # find index of depot (id==0)
    depot_idx = int(np.where(np.array(node_ids) == 0)[0][0]) if 0 in node_ids else 0
    customer_mask = [i for i in range(n_nodes) if i != depot_idx]
    if cluster_method == "KMeans (default)":
        if len(customer_mask) >= num_vehicles:
            # apply KMeans on customer coords
            kmeans = KMeans(n_clusters=num_vehicles, random_state=seed)
            cust_coords = coords_local[customer_mask]
            labels = kmeans.fit_predict(cust_coords)
            assignment = {v: [] for v in range(num_vehicles)}
            for idx, lab in zip(customer_mask, labels):
                assignment[lab].append(int(idx))
        else:
            # fewer customers than vehicles: assign one per vehicle until exhausted
            assignment = {v: [] for v in range(num_vehicles)}
            for i, idx in enumerate(customer_mask):
                assignment[i].append(int(idx))
    else:
        # greedy capacity-aware
        assignment = greedy_capacity_assign(coords_local, st.session_state['demands'], num_vehicles, vehicle_caps)

    st.session_state['cluster_assignment'] = assignment
else:
    # if not run, keep last assignment (or None)
    assignment = st.session_state.get('cluster_assignment', None)

# Show cluster table & plot cluster assignment
st.subheader("Cluster assignments (Stage 1 result)")
if assignment is None:
    st.info("No clustering performed yet. Click 'Run clustering and assign customers to vehicles' or 'Run full pipeline'.")
else:
    # Show table: customer id -> vehicle
    rows = []
    for v in range(num_vehicles):
        for nd in assignment.get(v, []):
            rows.append({"customer_index": int(nd), "id": int(node_ids[nd]), "vehicle": int(v), "demand": int(st.session_state['demands'][nd])})
    df_assign = pd.DataFrame(rows)
    if df_assign.empty:
        st.write("No customers assigned (check data).")
    else:
        st.dataframe(df_assign)

    # Plot clusters with Plotly
    fig_clusters = go.Figure()
    cmap = px_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    # depot
    dpt_x, dpt_y = coords[depot_idx]
    fig_clusters.add_trace(go.Scatter(x=[dpt_x], y=[dpt_y], mode='markers+text', marker=dict(size=16, color='gold', line=dict(width=1, color='black')), text=["Depot (0)"], textposition="top center", name="Depot"))
    for v in range(num_vehicles):
        custs = assignment.get(v, [])
        xs = coords[custs, 0] if custs else np.array([])
        ys = coords[custs, 1] if custs else np.array([])
        texts = [f"id:{int(node_ids[i])}<br>d:{int(st.session_state['demands'][i])}" for i in custs]
        if len(xs) > 0:
            fig_clusters.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', marker=dict(size=12, color=cmap[v % len(cmap)]), text=texts, textposition="bottom center", name=f"Vehicle {v} cluster"))
    fig_clusters.update_layout(title="Customer clusters assigned to vehicles (Stage 1)", height=450)
    st.plotly_chart(fig_clusters, use_container_width=True)

# -----------------------------
# Stage 2: Routing per cluster
# -----------------------------
st.subheader("5) Stage 2: Routing (per-cluster OR-Tools) and KPI computation")
if run_pipeline:
    # Prepare cluster-wise inputs and run VRP per cluster
    rng_global = np.random.default_rng(seed + 123)
    # Apply stochastic variation to base matrices as per sliders (one combined seed)
    var_d = demand_var_pct / 100.0
    var_t = travel_var_pct / 100.0
    # generate varied demands and distances
    varied_demands = st.session_state['demands'].copy().astype(int)
    varied_dist = apply_variation_matrix(st.session_state['dist_matrix'], var_t, rng_global)

    if var_d > 0:
        # only perturb customer demands, not depot (index where id==0)
        demand_noise = rng_global.uniform(1 - var_d, 1 + var_d, size=varied_demands.shape)
        varied_demands = np.round(varied_demands * demand_noise).astype(int)
        varied_demands[0] = 0  # depot demand zero

    # For each vehicle cluster, construct local matrix and solve
    routes_info_all = []
    cluster_caps = vehicle_caps.copy()
    for v in range(num_vehicles):
        custs = assignment.get(v, [])
        # local node ordering: [depot_idx] + custs
        local_indices = [depot_idx] + list(custs)
        if len(local_indices) <= 1:
            # no customers for this vehicle
            routes_info_all.append({"routes": [], "metrics": {"total_distance": 0, "served": 0, "total_load": 0}})
            continue
        # build local distance matrix
        local_n = len(local_indices)
        local_mat = np.zeros((local_n, local_n), dtype=int)
        local_dem = []
        for i_local, i_global in enumerate(local_indices):
            local_dem.append(int(varied_demands[i_global]))
            for j_local, j_global in enumerate(local_indices):
                local_mat[i_local, j_local] = int(varied_dist[i_global, j_global])
        # vehicle capacity for this vehicle
        cap = cluster_caps[v] if v < len(cluster_caps) else cluster_caps[-1]
        # Solve VRP for this cluster (treat single vehicle routing)
        routes_local, metrics_local = solve_vrp_ortools(local_mat, local_dem, cap, penalty=10000, time_limit=5)
        # Translate local node indices back to global indices in routes
        translated_routes = []
        for r in routes_local:
            translated = [local_indices[idx] for idx in r]
            translated_routes.append(translated)
        routes_info_all.append({"routes": translated_routes, "metrics": metrics_local})
    # compute KPIs
    kpis = compute_kpis(routes_info_all, assignment, None, cluster_caps, None)

    st.subheader("KPI Dashboard (after stochastic variation & routing)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Distance", f"{kpis['total_distance']}")
    c2.metric("Total Assigned Load", f"{kpis['total_load']}")
    c3.metric("Vehicles used", f"{kpis['vehicles_used']}/{num_vehicles}")
    c4.metric("Avg Utilization (%)", f"{kpis['avg_utilization_pct']:.1f}%")

    # Show per-vehicle route summary
    rows = []
    for vidx, info in enumerate(routes_info_all):
        routes_local = info.get('routes', [])
        met = info.get('metrics', {})
        if not routes_local:
            rows.append({"Vehicle": vidx, "Route": "-", "Distance": met.get('total_distance', 0), "Load": met.get('total_load', 0)})
        else:
            for r in routes_local:
                rows.append({"Vehicle": vidx, "Route": "→".join(map(str, r)), "Distance": met.get('total_distance', 0), "Load": met.get('total_load', 0)})
    df_routes_summary = pd.DataFrame(rows)
    st.dataframe(df_routes_summary)

    # Plot full route map with Plotly
    st.subheader("Route Visualization (Plotly)")
    fig = go.Figure()
    # plot all nodes
    fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='markers+text', marker=dict(size=10, color='lightgray'), text=[f"id:{int(i)}" for i in node_ids], textposition="bottom center", name="Nodes"))
    # depot highlight
    fig.add_trace(go.Scatter(x=[coords[depot_idx, 0]], y=[coords[depot_idx, 1]], mode='markers+text', marker=dict(size=18, color='gold', line=dict(width=2)), text=["Depot (0)"], textposition="top center", name="Depot"))
    # draw per-vehicle routes
    for vidx, info in enumerate(routes_info_all):
        color = cmap[vidx % len(cmap)]
        for r in info.get('routes', []):
            # r is list of global indices
            xs = [coords[int(idx), 0] for idx in r]
            ys = [coords[int(idx), 1] for idx in r]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=3, color=color), marker=dict(size=8), name=f"Vehicle {vidx} route"))
    fig.update_layout(title="Routing result after clustering + per-cluster VRP", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Offer CSV download of results
    csv_buf = io.StringIO()
    df_routes_summary.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    st.download_button("Download routes CSV", data=csv_bytes, file_name="optiroute_routes.csv", mime="text/csv")

    # Sensitivity quick check: vary several seeds and show change % in total distance
    st.subheader("Sensitivity quick-scan (multiple random draws)")
    samples = st.number_input("Number of stochastic samples for quick-scan", min_value=1, max_value=100, value=10)
    if st.button("Run sensitivity quick-scan"):
        base_total = kpis['total_distance']
        rng_scan = np.random.default_rng(seed + 999)
        sample_totals = []
        for s in range(int(samples)):
            # vary both demand and travel times
            vard = rng_scan.uniform(1 - var_d, 1 + var_d, size=st.session_state['demands'].shape)
            demand_s = np.round(st.session_state['demands'] * vard).astype(int)
            demand_s[0] = 0
            dist_s = apply_variation_matrix(st.session_state['dist_matrix'], var_t, rng_scan)
            # re-run clustering assignment -> routing quickly using same assignment (we perturb and reuse assignment)
            # For simplicity, we will solve per cluster like before
            routes_info_temp = []
            for v in range(num_vehicles):
                custs = assignment.get(v, [])
                local_indices = [depot_idx] + list(custs)
                if len(local_indices) <= 1:
                    routes_info_temp.append({"metrics": {"total_distance": 0, "served": 0, "total_load": 0}})
                    continue
                local_mat = np.zeros((len(local_indices), len(local_indices)), dtype=int)
                local_dem = []
                for i_local, i_global in enumerate(local_indices):
                    local_dem.append(int(demand_s[i_global]))
                    for j_local, j_global in enumerate(local_indices):
                        local_mat[i_local, j_local] = int(dist_s[i_global, j_global])
                cap = cluster_caps[v] if v < len(cluster_caps) else cluster_caps[-1]
                _, metrics_tmp = solve_vrp_ortools(local_mat, local_dem, cap, penalty=10000, time_limit=2)
                routes_info_temp.append({"metrics": metrics_tmp})
            kpi_tmp = compute_kpis(routes_info_temp, assignment, None, cluster_caps, None)
            sample_totals.append(kpi_tmp['total_distance'])
        # plot sample totals
        sample_arr = np.array(sample_totals)
        pct_change = (sample_arr - base_total) / base_total * 100 if base_total > 0 else np.zeros_like(sample_arr)
        st.line_chart(pd.DataFrame({"Total distance samples": sample_arr, "Pct change vs base": pct_change}))
        st.write("Sensitivity summary (samples):", f"mean change % = {pct_change.mean():.2f}", f"std % = {pct_change.std():.2f}")

else:
    st.info("Press 'Run full pipeline' to perform clustering → routing → KPI calculation under current stochastic settings.")

# End of app
st.markdown("---")
st.caption("OptiRoute — Decomposed VRP with Stochastic Sensitivity & KPI Dashboard. Built for MSL304, IIT Delhi.")
