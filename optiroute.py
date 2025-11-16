import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from itertools import permutations

# ======================================================
# Helper functions
# ======================================================

def create_distance_matrix(points):
    """Compute Euclidean distance matrix."""
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            dist[i, j] = dist[j, i] = d
    return dist


def solve_vrp(dist_matrix, demand, capacities):
    """Simple brute-force VRP solver for small sets."""
    n = len(demand)
    best_order = None
    best_distance = np.inf

    for perm in permutations(range(1, n)):  # exclude depot (0)
        order = [0] + list(perm) + [0]
        total = 0
        valid = True
        load = 0
        for i in range(len(order) - 1):
            load += demand[order[i]]
            if load > capacities[0]:
                valid = False
                break
            total += dist_matrix[order[i], order[i + 1]]
        if valid and total < best_distance:
            best_distance = total
            best_order = order
    return best_order, best_distance


# ======================================================
# Streamlit App
# ======================================================

st.set_page_config(page_title="OptiRoute 🚚", layout="wide")
st.title("🚚 OptiRoute: Route Optimization & Sensitivity Analysis")

st.sidebar.header("📦 Input Configuration")
num_customers = st.sidebar.number_input("Number of customers", 5, 50, 8)
num_vehicles = st.sidebar.number_input("Number of vehicles", 1, 10, 3)
vehicle_capacity = st.sidebar.number_input("Vehicle capacity", 10, 1000, 100)
np.random.seed(st.sidebar.number_input("Random seed", 0, 9999, 42))

st.write("---")

# ======================================================
# Stage 1: Data Generation or Upload
# ======================================================
st.subheader("📍 Stage 1: Generate or Upload Data")

data_option = st.radio("Choose input method:", ["Generate Random Data", "Upload CSV"])

if data_option == "Generate Random Data":
    x = np.random.uniform(0, 100, num_customers)
    y = np.random.uniform(0, 100, num_customers)
    demand = np.random.randint(1, 20, num_customers)
    data = pd.DataFrame({"x": x, "y": y, "demand": demand})
    st.session_state.data = data
    st.map(data.rename(columns={"x": "lat", "y": "lon"}))
else:
    uploaded_file = st.file_uploader("Upload a CSV with x, y, demand columns", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.dataframe(data)

if "data" not in st.session_state:
    st.stop()

st.write("---")

# ======================================================
# Stage 2: Optimization
# ======================================================

st.subheader("⚙️ Stage 2: Optimization")

if st.button("🚀 Run Optimization"):
    st.session_state.optimized_data = st.session_state.data.copy()

    data = st.session_state.optimized_data
    km = KMeans(n_clusters=num_vehicles, n_init=10, random_state=42)
    customers = data.copy()
    customers["cluster"] = km.fit_predict(customers[["x", "y"]])

    total_distance = 0
    for c in range(num_vehicles):
        cluster_points = customers[customers["cluster"] == c]
        depot = pd.DataFrame({"x": [50], "y": [50], "demand": [0]})
        cluster_points = pd.concat([depot, cluster_points], ignore_index=True)
        dist_matrix = create_distance_matrix(cluster_points[["x", "y"]].values)
        _, dist = solve_vrp(dist_matrix, cluster_points["demand"].to_list(), [vehicle_capacity])
        total_distance += dist

    st.session_state.optimized_total_distance = total_distance
    st.session_state.optimized_clusters = customers

st.write("### Optimization Results")
if "optimized_clusters" in st.session_state:
    customers = st.session_state.optimized_clusters
    fig = go.Figure()
    for c in range(num_vehicles):
        subset = customers[customers["cluster"] == c]
        fig.add_trace(go.Scatter(
            x=subset["x"], y=subset["y"], mode="markers+lines",
            name=f"Vehicle {c+1}"
        ))
    fig.update_layout(title="Optimized Routes", xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Total Optimized Distance", f"{st.session_state.optimized_total_distance:.2f}")

st.write("---")

# ======================================================
# Stage 3: Sensitivity Quick Scan
# ======================================================

st.subheader("🎯 Stage 3: Sensitivity Quick Scan")

if "optimized_data" not in st.session_state:
    st.session_state.optimized_data = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

num_samples = st.number_input("Number of stochastic samples", 2, 20, 5)
scan_button = st.button("🔁 Run Sensitivity Quick Scan")

if scan_button:
    if st.session_state.optimized_data is None:
        st.warning("⚠️ Please run the optimization first before running the sensitivity scan.")
        st.stop()

    st.session_state.scan_results = []  # reset results
    progress = st.progress(0)
    status = st.empty()

    for i in range(int(num_samples)):
        status.text(f"Running sample {i+1}/{num_samples}...")
        noisy_data = st.session_state.optimized_data.copy()
        noisy_data["demand"] *= np.random.uniform(0.9, 1.1, len(noisy_data))
        noisy_data[["x", "y"]] += np.random.normal(0, 2, size=(len(noisy_data), 2))

        km = KMeans(n_clusters=num_vehicles, n_init=10, random_state=42 + i)
        customers = noisy_data.copy()
        customers["cluster"] = km.fit_predict(customers[["x", "y"]])

        total_distance = 0
        for c in range(num_vehicles):
            cluster_points = customers[customers["cluster"] == c]
            depot = pd.DataFrame({"x": [50], "y": [50], "demand": [0]})
            cluster_points = pd.concat([depot, cluster_points], ignore_index=True)
            dist_matrix = create_distance_matrix(cluster_points[["x", "y"]].values)
            _, dist = solve_vrp(dist_matrix, cluster_points["demand"].to_list(), [vehicle_capacity])
            total_distance += dist

        st.session_state.scan_results.append(total_distance)
        progress.progress((i + 1) / num_samples)

    status.text("✅ Scan complete!")

if st.session_state.scan_results:
    df_results = pd.DataFrame({
        "Run": np.arange(1, len(st.session_state.scan_results) + 1),
        "Total Distance": st.session_state.scan_results
    })
    st.dataframe(df_results)

    fig = go.Figure(data=[go.Bar(x=df_results["Run"], y=df_results["Total Distance"])])
    fig.update_layout(title="Total Distance per Random Run", xaxis_title="Run", yaxis_title="Distance")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Average Distance", f"{np.mean(st.session_state.scan_results):.2f}")
    st.metric("Std. Dev.", f"{np.std(st.session_state.scan_results):.2f}")
