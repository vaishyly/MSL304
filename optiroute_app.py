# ---------------- Part 1/4 ----------------
"""
OptiRoute - Full Streamlit VRP app (Multi-objective + Time Windows + Emissions + Lateness KPI)
Copy parts 1..4 into app.py in order.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# ---------------------------
# Helper functions
# ---------------------------
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

def apply_variation_matrix(base_matrix, var_pct, rng):
    if var_pct <= 0:
        return base_matrix.copy()
    noise = rng.uniform(1 - var_pct, 1 + var_pct, size=base_matrix.shape)
    varied = np.round(base_matrix * noise).astype(int)
    varied[base_matrix == 0] = 0
    np.fill_diagonal(varied, 0)
    return varied

# ---------------------------
# OR-Tools solver for a single cluster (with time windows)
# ---------------------------
def solve_vrp_ortools(dist_matrix, demands, service_times, time_windows, vehicle_capacity,
                      weighted_cost_factor=1.0, penalty=10000, time_limit=10):
    """
    dist_matrix: numpy (n x n) ints
    demands, service_times: lists length n (index 0 is depot)
    time_windows: list of (start,end) length n
    vehicle_capacity: int
    weighted_cost_factor: multiplier used inside arc-cost callback
    """
    n = len(dist_matrix)
    if n == 0:
        return [], {"total_distance": 0, "served": 0, "total_load": 0, "arrival_times": {}, "total_lateness": 0}
    if n == 1:
        return [[0, 0]], {"total_distance": 0, "served": 0, "total_load": 0, "arrival_times": {}, "total_lateness": 0}

    depot = 0
    num_vehicles = 1
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Base travel time/distance callback
    def travel_cb(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])

    travel_idx = routing.RegisterTransitCallback(travel_cb)

    # Weighted cost callback (for objective)
    def weighted_cost_cb(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        base = dist_matrix[from_node][to_node]
        return int(round(base * weighted_cost_factor))

    weighted_idx = routing.RegisterTransitCallback(weighted_cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(weighted_idx)

    # Demand callback and capacity dimension
    def demand_cb(from_index):
        node = manager.IndexToNode(from_index)
        return int(demands[node])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [vehicle_capacity], True, "Capacity")

    # Time callback: travel time + service at from node
    def time_cb(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = int(dist_matrix[from_node][to_node])
        serv = int(service_times[from_node])
        return travel + serv

    time_idx = routing.RegisterTransitCallback(time_cb)
    horizon = 24 * 60 * 60
    routing.AddDimension(time_idx, 1000000, horizon, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Add time windows as hard constraints
    for node in range(n):
        index = manager.NodeToIndex(node)
        tw_start, tw_end = int(time_windows[node][0]), int(time_windows[node][1])
        # Guard: if tw_start > tw_end, swap / expand
        if tw_start > tw_end:
            tw_end = tw_start + 1
        time_dim.CumulVar(index).SetRange(tw_start, tw_end)

    # Allow drop with penalty
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = int(time_limit)
    search_params.log_search = False

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return [], {"total_distance": None, "served": 0, "total_load": 0, "arrival_times": {}, "total_lateness": None}

    routes = []
    total_distance = 0
    total_load = 0
    served = 0
    arrival_times = {}
    total_lateness = 0

    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        route_dist = 0
        route_load = 0
        prev_index = None
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            route_load += int(demands[node])
            tvar = time_dim.CumulVar(index)
            arrival = solution.Value(tvar)
            arrival_times[node] = arrival
            tw_end = int(time_windows[node][1])
            if arrival > tw_end:
                total_lateness += (arrival - tw_end)
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                curr = manager.IndexToNode(prev_index)
                nxt = manager.IndexToNode(index)
                route_dist += int(dist_matrix[curr][nxt])
        route.append(depot)
        served_local = sum(1 for nd in route if nd != depot)
        if len(route) > 2:
            routes.append(route)
            total_distance += route_dist
            total_load += route_load
            served += served_local

    metrics = {
        "total_distance": total_distance,
        "served": served,
        "total_load": total_load,
        "arrival_times": arrival_times,
        "total_lateness": total_lateness
    }
    return routes, metrics

# ---------------------------
# KPI computation
# ---------------------------
def compute_kpis(routes_all, cluster_caps, fuel_cost_per_km, emission_factor, alpha, beta, gamma):
    total_distance = 0
    total_load = 0
    utilizations = []
    served_count = 0
    total_lateness = 0

    for vidx, rinfo in enumerate(routes_all):
        if rinfo is None or 'metrics' not in rinfo:
            utilizations.append(0.0)
            continue
        metrics = rinfo['metrics']
        td = metrics.get('total_distance', 0) or 0
        tload = metrics.get('total_load', 0) or 0
        laten = metrics.get('total_lateness', 0) or 0
        total_distance += td
        total_load += tload
        total_lateness += laten
        cap = cluster_caps[vidx] if vidx < len(cluster_caps) else cluster_caps[-1]
        utilization = 100.0 * tload / cap if cap > 0 else 0.0
        utilizations.append(utilization)
        served_count += metrics.get('served', 0) or 0

    total_cost = total_distance * fuel_cost_per_km
    total_emission = total_distance * emission_factor
    objective_score = alpha * total_cost + beta * total_emission + gamma * total_lateness

    avg_util = np.mean(utilizations) if len(utilizations) > 0 else 0
    kpis = {
        "total_distance": total_distance,
        "total_load": total_load,
        "vehicles_used": sum(1 for u in utilizations if u > 0),
        "avg_utilization_pct": avg_util,
        "served_count": served_count,
        "total_cost": total_cost,
        "total_emission": total_emission,
        "total_lateness": total_lateness,
        "objective_score": objective_score
    }
    return kpis
# ---------------- end Part 1/4 ----------------


# ---------------- Part 2/4 ----------------
# Streamlit UI & initial data (Part 2/4)

st.set_page_config(page_title="OptiRoute — VRP (Multi-objective + Time Windows)", layout="wide")
st.title("OptiRoute — Multi-objective VRP (Cost, Emissions, Time Windows & KPI Dashboard)")
st.markdown("Team: Vaishali Anand, Arpit Agrawal, Vipul Yadav — IIT Delhi, MSL304")

# Sidebar: Demo options, weights, cost/emission params, CSV upload
st.sidebar.header("Demo / Data options")
demo_customers = st.sidebar.number_input("Demo customers (excluding depot)", min_value=2, max_value=50, value=8, step=1)
demo_vehicles = st.sidebar.number_input("Demo number of vehicles", min_value=1, max_value=6, value=3, step=1)
seed = st.sidebar.number_input("Random seed (demo)", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Multi-objective weights")
alpha = st.sidebar.slider("α: Cost weight", 0.0, 5.0, 1.0)
beta  = st.sidebar.slider("β: Emission weight", 0.0, 5.0, 1.0)
gamma = st.sidebar.slider("γ: Lateness weight", 0.0, 5.0, 1.0)

st.sidebar.subheader("Cost & Emission params")
fuel_cost_per_km = st.sidebar.number_input("Fuel cost per km (₹ or unit)", value=5.0, min_value=0.0)
emission_factor = st.sidebar.number_input("Emission per km (kg CO2 or unit)", value=0.2, min_value=0.0)

st.sidebar.markdown("---")
st.sidebar.write("CSV columns allowed: id,x,y,demand,service_time,tw_start,tw_end (id should include depot as 0).")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# Create or load base dataframe with time windows & service times
if 'base_df' not in st.session_state or st.session_state.get("last_demo_params", None) != (demo_customers, demo_vehicles, seed):
    rng = np.random.default_rng(seed)
    depot = {"id": 0, "x": 50.0, "y": 50.0, "demand": 0, "service_time": 0, "tw_start": 0, "tw_end": 240}
    customers = []
    for i in range(1, demo_customers + 1):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(5, 40)
        x = depot['x'] + r * np.cos(angle)
        y = depot['y'] + r * np.sin(angle)
        demand = int(rng.integers(2, 30))
        tw_center = int(rng.integers(30, 180))
        tw_width = int(rng.integers(20, 80))
        service_time = int(rng.integers(2, 10))
        customers.append({"id": i, "x": float(x), "y": float(y), "demand": int(demand),
                          "service_time": int(service_time),
                          "tw_start": max(0, tw_center - tw_width // 2),
                          "tw_end": tw_center + tw_width // 2})
    df_demo = pd.DataFrame([depot] + customers)
    st.session_state['base_df'] = df_demo
    st.session_state['last_demo_params'] = (demo_customers, demo_vehicles, seed)

# If CSV uploaded, validate and use
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        expected_cols = {"id", "x", "y", "demand", "service_time", "tw_start", "tw_end"}
        if not expected_cols.issubset(set(uploaded_df.columns)):
            st.sidebar.error(f"CSV must contain columns: {', '.join(sorted(expected_cols))}")
        else:
            st.session_state['base_df'] = uploaded_df.copy()
    except Exception as e:
        st.sidebar.error(f"Failed to parse CSV: {e}")

# Editable nodes table
st.subheader("1) Locations, Demands & Time Windows")
df_nodes = st.data_editor(st.session_state['base_df'], num_rows="dynamic", key="nodes_editor")

# Validate id/depot
if 'id' not in df_nodes.columns:
    st.error("Data must include 'id' column. Edit to include id (0 is depot).")
    st.stop()

if 0 not in df_nodes['id'].values:
    st.warning("No depot with id 0 found. Treating first row as depot and assigning id 0.")
    df_nodes.loc[df_nodes.index[0], 'id'] = 0

# Normalize types & build matrices
nodes = df_nodes.sort_values('id').reset_index(drop=True)
coords = nodes[['x', 'y']].to_numpy()
demands = nodes['demand'].astype(int).to_numpy()
service_times = nodes['service_time'].astype(int).to_numpy()
time_windows = list(zip(nodes['tw_start'].astype(int).to_list(), nodes['tw_end'].astype(int).to_list()))
node_ids = nodes['id'].astype(int).to_list()
depot_idx = int(np.where(np.array(node_ids) == 0)[0][0]) if 0 in node_ids else 0

base_dist_matrix = euclid_matrix(coords)

# Vehicle params
st.subheader("2) Vehicle Parameters")
num_vehicles = st.number_input("Number of vehicles", min_value=1, value=demo_vehicles, step=1)
vehicle_capacity_default = st.number_input("Default vehicle capacity", min_value=1, value=100, step=1)
vehicle_caps = []
vehicle_maxdist = []
vehicle_cols = st.columns(min(4, num_vehicles))
for i in range(num_vehicles):
    with vehicle_cols[i % len(vehicle_cols)]:
        cap = st.number_input(f"Vehicle {i} capacity", value=vehicle_capacity_default, min_value=1, key=f'cap_{i}')
        md = st.number_input(f"Vehicle {i} max distance", value=1000, min_value=1, key=f'md_{i}')
    vehicle_caps.append(int(cap))
    vehicle_maxdist.append(int(md))

# Clustering & stochastic controls
st.subheader("3) Stage 1: Customer assignment (Clustering)")
cluster_method = st.selectbox("Clustering method", options=["KMeans (default)", "Greedy capacity-aware"], index=0)
do_cluster = st.button("Run clustering and assign customers to vehicles")

st.subheader("4) Stochastic Sensitivity (What-if)")
col_a, col_b = st.columns(2)
with col_a:
    demand_var_pct = st.slider("Demand variation ±% (simulate)", min_value=0, max_value=50, value=10, step=1)
with col_b:
    travel_var_pct = st.slider("Travel time/distance variation ±% (simulate)", min_value=0, max_value=50, value=10, step=1)

st.markdown("Click 'Run full pipeline' to perform clustering → routing → KPIs under current stochastic settings.")
run_pipeline = st.button("Run full pipeline (clustering → routing → KPIs)")

# Save into session
st.session_state['base_df'] = nodes
st.session_state['coords'] = coords
st.session_state['demands'] = demands
st.session_state['service_times'] = service_times
st.session_state['time_windows'] = time_windows
st.session_state['dist_matrix'] = base_dist_matrix

# Greedy capacity assign
def greedy_capacity_assign(coords, demands, num_vehicles, vehicle_caps):
    n = coords.shape[0]
    customers = [i for i in range(n) if i != 0]
    remaining = {v: vehicle_caps[v] for v in range(num_vehicles)}
    assignment = {v: [] for v in range(num_vehicles)}
    customers_sorted = sorted(customers, key=lambda i: demands[i], reverse=True)
    for c in customers_sorted:
        best_v = max(range(num_vehicles), key=lambda v: remaining[v])
        assignment[best_v].append(c)
        remaining[best_v] -= demands[c]
    return assignment

# Clustering execution
if 'cluster_assignment' not in st.session_state:
    st.session_state['cluster_assignment'] = None

if do_cluster or run_pipeline:
    coords_local = st.session_state['coords']
    n_nodes = coords_local.shape[0]
    customer_mask = [i for i in range(n_nodes) if i != depot_idx]
    if cluster_method == "KMeans (default)":
        if len(customer_mask) >= num_vehicles:
            kmeans = KMeans(n_clusters=num_vehicles, random_state=seed)
            cust_coords = coords_local[customer_mask]
            labels = kmeans.fit_predict(cust_coords)
            assignment = {v: [] for v in range(num_vehicles)}
            for idx, lab in zip(customer_mask, labels):
                assignment[lab].append(int(idx))
        else:
            assignment = {v: [] for v in range(num_vehicles)}
            for i, idx in enumerate(customer_mask):
                assignment[i].append(int(idx))
    else:
        assignment = greedy_capacity_assign(coords_local, st.session_state['demands'], num_vehicles, vehicle_caps)

    st.session_state['cluster_assignment'] = assignment
else:
    assignment = st.session_state.get('cluster_assignment', None)

# Show cluster assignment & plot
st.subheader("Cluster assignments (Stage 1 result)")
if assignment is None:
    st.info("No clustering performed yet. Click 'Run clustering and assign customers to vehicles' or 'Run full pipeline'.")
else:
    rows = []
    for v in range(num_vehicles):
        for nd in assignment.get(v, []):
            rows.append({"customer_index": int(nd), "id": int(node_ids[nd]), "vehicle": int(v), "demand": int(st.session_state['demands'][nd])})
    df_assign = pd.DataFrame(rows)
    if df_assign.empty:
        st.write("No customers assigned (check data).")
    else:
        st.dataframe(df_assign)

    # Plot clusters
    fig_clusters = go.Figure()
    cmap = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    dpt_x, dpt_y = coords[depot_idx]
    fig_clusters.add_trace(go.Scatter(x=[dpt_x], y=[dpt_y], mode='markers+text',
                                      marker=dict(size=16, color='gold', line=dict(width=1, color='black')),
                                      text=["Depot (0)"], textposition="top center", name="Depot"))
    for v in range(num_vehicles):
        custs = assignment.get(v, [])
        xs = coords[custs,0] if custs else np.array([])
        ys = coords[custs,1] if custs else np.array([])
        texts = []
        for i in custs:
            tw = st.session_state['time_windows'][i]
            texts.append(f"id:{int(node_ids[i])}<br>d:{int(st.session_state['demands'][i])}<br>tw:{tw[0]}-{tw[1]}")
        if len(xs)>0:
            fig_clusters.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text',
                                              marker=dict(size=12, color=cmap[v%len(cmap)]),
                                              text=texts, textposition="bottom center", name=f"Vehicle {v} cluster"))
    fig_clusters.update_layout(title="Customer clusters assigned to vehicles (Stage 1)", height=450)
    st.plotly_chart(fig_clusters, use_container_width=True)

# ---------------- end Part 2/4 ----------------


# ---------------- Part 3/4 ----------------
# Stage 2: routing + KPIs (Part 3/4)

st.subheader("5) Stage 2: Routing (per-cluster OR-Tools) and KPI computation")

if run_pipeline:
    rng_global = np.random.default_rng(seed + 123)
    var_d = demand_var_pct / 100.0
    var_t = travel_var_pct / 100.0

    varied_demands = st.session_state['demands'].copy().astype(int)
    varied_dist = apply_variation_matrix(st.session_state['dist_matrix'], var_t, rng_global)

    if var_d > 0:
        demand_noise = rng_global.uniform(1 - var_d, 1 + var_d, size=varied_demands.shape)
        varied_demands = np.round(varied_demands * demand_noise).astype(int)
        varied_demands[depot_idx] = 0

    # weighted cost factor for OR-Tools arc cost: combine alpha & beta as a scalar
    weighted_cost_factor = alpha * fuel_cost_per_km + beta * emission_factor

    routes_info_all = []
    cluster_caps = vehicle_caps.copy()
    for v in range(num_vehicles):
        custs = assignment.get(v, [])
        local_indices = [depot_idx] + list(custs)
        if len(local_indices) <= 1:
            routes_info_all.append({"routes": [], "metrics": {"total_distance": 0, "served": 0, "total_load": 0, "arrival_times": {}, "total_lateness": 0}})
            continue

        local_n = len(local_indices)
        local_mat = np.zeros((local_n, local_n), dtype=int)
        local_dem = []
        local_serv = []
        local_tw = []
        for i_local, i_global in enumerate(local_indices):
            local_dem.append(int(varied_demands[i_global]))
            local_serv.append(int(st.session_state['service_times'][i_global]))
            local_tw.append((int(st.session_state['time_windows'][i_global][0]), int(st.session_state['time_windows'][i_global][1])))
            for j_local, j_global in enumerate(local_indices):
                local_mat[i_local, j_local] = int(varied_dist[i_global, j_global])

        cap = cluster_caps[v] if v < len(cluster_caps) else cluster_caps[-1]
        routes_local, metrics_local = solve_vrp_ortools(local_mat, local_dem, local_serv,
                                                       local_tw, cap,
                                                       weighted_cost_factor=weighted_cost_factor,
                                                       penalty=10000, time_limit=6)
        # Translate local indices back to global indices
        translated = []
        for r in routes_local:
            translated.append([local_indices[idx] for idx in r])
        routes_info_all.append({"routes": translated, "metrics": metrics_local})

    # Compute KPIs
    kpis = compute_kpis(routes_info_all, cluster_caps, fuel_cost_per_km, emission_factor, alpha, beta, gamma)

    # Per-vehicle route summary (table) - we will show KPI block just after this table (user requested B)
    rows = []
    for vidx, info in enumerate(routes_info_all):
        routes_local = info.get('routes', [])
        met = info.get('metrics', {})
        if not routes_local:
            rows.append({"Vehicle": vidx, "Route": "-", "Distance": met.get('total_distance', 0), "Load": met.get('total_load', 0), "Lateness": met.get('total_lateness', 0)})
        else:
            for r in routes_local:
                rows.append({"Vehicle": vidx, "Route": "->".join(map(str, r)), "Distance": met.get('total_distance', 0), "Load": met.get('total_load', 0), "Lateness": met.get('total_lateness', 0)})
    df_routes_summary = pd.DataFrame(rows)
    st.dataframe(df_routes_summary)
# else: if not run_pipeline, we don't show routing / KPIs
# ---------------- end Part 3/4 ----------------


# ---------------- Part 4/4 ----------------
# KPI Dashboard (placed BELOW the routes table as requested)

if run_pipeline:
    st.subheader("📊 KPI Dashboard — After Routing")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Distance (km)", f"{kpis['total_distance']}")
    k2.metric("Total Delivered Load", f"{kpis['total_load']}")
    k3.metric("Vehicles Used", f"{kpis['vehicles_used']}/{num_vehicles}")
    k4.metric("Avg Utilization (%)", f"{kpis['avg_utilization_pct']:.1f}%")
    k5.metric("Total Lateness (units)", f"{kpis['total_lateness']}")

    k6, k7, k8 = st.columns(3)
    k6.metric("Total Fuel Cost", f"{kpis['total_cost']:.2f}")
    k7.metric("Total Emissions", f"{kpis['total_emission']:.2f}")
    k8.metric("Objective Score", f"{kpis['objective_score']:.2f}")

    st.write("### 🔍 Objective Function")
    st.write(f"""
    **Z = α·Cost + β·Emission + γ·Lateness**  
    Where:  
    - α = {alpha}  
    - β = {beta}  
    - γ = {gamma}  
    ---
    """)

    # Route visualization
    st.subheader("Route Visualization")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode='markers+text',
                             marker=dict(size=10, color='lightgray'),
                             text=[f"id:{int(i)}" for i in node_ids], textposition="bottom center", name="Nodes"))
    fig.add_trace(go.Scatter(x=[coords[depot_idx, 0]], y=[coords[depot_idx, 1]], mode='markers+text',
                             marker=dict(size=18, color='gold', line=dict(width=2)),
                             text=["Depot (0)"], textposition="top center", name="Depot"))
    cmap = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    for vidx, info in enumerate(routes_info_all):
        color = cmap[vidx % len(cmap)]
        for r in info.get('routes', []):
            xs = [coords[int(idx), 0] for idx in r]
            ys = [coords[int(idx), 1] for idx in r]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(width=3, color=color),
                                     marker=dict(size=8), name=f"Vehicle {vidx} route"))
    fig.update_layout(title="Routing result after clustering + per-cluster VRP", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Download CSV
    csv_buf = io.StringIO()
    df_routes_summary.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    st.download_button("Download routes CSV", data=csv_bytes, file_name="optiroute_routes.csv", mime="text/csv")

    # Sensitivity quick-scan
    st.subheader("Sensitivity quick-scan (multiple random draws)")
    samples = st.number_input("Number of stochastic samples for quick-scan", min_value=1, max_value=100, value=10)
    if st.button("Run sensitivity quick-scan"):
        base_total = kpis['total_distance'] if kpis['total_distance'] is not None else 0
        rng_scan = np.random.default_rng(seed + 999)
        sample_totals = []
        for s in range(int(samples)):
            vard = rng_scan.uniform(1 - var_d, 1 + var_d, size=st.session_state['demands'].shape)
            demand_s = np.round(st.session_state['demands'] * vard).astype(int)
            demand_s[depot_idx] = 0
            dist_s = apply_variation_matrix(st.session_state['dist_matrix'], var_t, rng_scan)
            routes_info_temp = []
            for v in range(num_vehicles):
                custs = assignment.get(v, [])
                local_indices = [depot_idx] + list(custs)
                if len(local_indices) <= 1:
                    routes_info_temp.append({"metrics": {"total_distance": 0, "served": 0, "total_load": 0, "total_lateness": 0}})
                    continue
                local_mat = np.zeros((len(local_indices), len(local_indices)), dtype=int)
                local_dem = []
                local_serv = []
                local_tw = []
                for i_local, i_global in enumerate(local_indices):
                    local_dem.append(int(demand_s[i_global]))
                    local_serv.append(int(st.session_state['service_times'][i_global]))
                    local_tw.append((int(st.session_state['time_windows'][i_global][0]), int(st.session_state['time_windows'][i_global][1])))
                    for j_local, j_global in enumerate(local_indices):
                        local_mat[i_local, j_local] = int(dist_s[i_global, j_global])
                cap = cluster_caps[v] if v < len(cluster_caps) else cluster_caps[-1]
                _, metrics_tmp = solve_vrp_ortools(
                    local_mat,
                    local_dem,
                    local_serv,
                    local_tw,
                    cap,
                    weighted_cost_factor=weighted_cost_factor,
                    penalty=10000,
                    time_limit=2
                )
                routes_info_temp.append({"metrics": metrics_tmp})
            kpi_tmp = compute_kpis(routes_info_temp, cluster_caps, fuel_cost_per_km, emission_factor, alpha, beta, gamma)
            sample_totals.append(kpi_tmp['total_distance'])
        sample_arr = np.array(sample_totals)
        pct_change = (sample_arr - base_total) / base_total * 100 if base_total > 0 else np.zeros_like(sample_arr)
        st.line_chart(pd.DataFrame({"Total distance samples": sample_arr, "Pct change vs base": pct_change}))
        st.write("Sensitivity summary (samples):", f"mean change % = {pct_change.mean():.2f}", f"std % = {pct_change.std():.2f}")

else:
    st.info("Press 'Run full pipeline' to perform clustering → routing → KPI calculation under current stochastic settings.")

st.markdown("---")
st.caption("OptiRoute — Multi-objective VRP with Time Windows & KPI Dashboard. Built for MSL304, IIT Delhi.")
# ---------------- end Part 4/4 ----------------
