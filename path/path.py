
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from networkx.exception import NetworkXNoPath
import math

# ---------------------------
# 1. Build full and truck-suitable networks
# ---------------------------
place = "Vilnius, Lithuania"
# ---------------------------
bins = [(54.6938,	25.2833),
(54.6938,	25.2833),
(54.6938,	25.2833),
(54.6938,	25.2833),
(54.6938,	25.2833),
(54.6899,	25.2509),
(54.6913,	25.2882),
(54.6894,	25.2775),
(54.6894,	25.2775),
(54.6956,	25.2368),
(54.6885,	25.2784),
(54.6885,	25.2784),
(54.6885,	25.2784),
(54.6885,	25.2784),
(54.6875,	25.274)
]
starts_4 = [(54.7097628, 25.1719080), (54.7262582, 25.2116795), (54.6073744, 25.1619865), (54.6374971, 25.3018982), (54.6911549, 25.4664075)]
sink = (54.66669207980455, 25.14615799325379)
K = 5  # number of trucks (adjust as needed)
capacity_nodes = 20 


all_points = bins + starts_4 + [sink]
n_bins = len(bins)
n_starts = len(starts_4)

# quick feasibility check
total_capacity = capacity_nodes * K
if n_bins > total_capacity:
    raise SystemExit(f"Not enough total capacity: {n_bins} bins > {total_capacity} total (capacity_nodes * K). "
                     f"Increase capacity_nodes or K, or reduce bins.")

print("Downloading OSM graph (may take a minute)...")
G_full = ox.graph_from_place(place, network_type="drive")
G_full.remove_nodes_from(list(nx.isolates(G_full)))

# prepare truck graph by filtering (we include residential/service to help connectivity)
G_truck = G_full.copy()
TRUCK_OK = {"motorway","motorway_link","trunk","trunk_link","primary","primary_link",
            "secondary","secondary_link","tertiary","tertiary_link","unclassified","residential","service"}
edges_to_remove = []
for u, v, k, data in G_truck.edges(keys=True, data=True):
    hw = data.get("highway")
    if isinstance(hw, list): hw = hw[0]
    if hw not in TRUCK_OK:
        edges_to_remove.append((u, v, k))
G_truck.remove_edges_from(edges_to_remove)
G_truck.remove_nodes_from(list(nx.isolates(G_truck)))

# add travel_time attribute
SPEEDS = {"motorway":80,"motorway_link":60,"trunk":70,"trunk_link":50,"primary":60,"primary_link":50,
          "secondary":50,"secondary_link":40,"tertiary":40,"tertiary_link":35,"unclassified":35,
          "residential":35, "service":35}
def add_travel_time_attr(G):
    for u,v,k,data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        if isinstance(hw, list): hw = hw[0]
        speed = SPEEDS.get(hw, 35)
        length = data.get("length", 0)
        data["travel_time"] = length / (max(speed,1) * 1000.0/3600.0)
add_travel_time_attr(G_full)
add_travel_time_attr(G_truck)

# robust snapping
def snap_to_node(G, latlon):
    lat, lon = latlon
    try:
        return ox.get_nearest_node(G, (lat, lon))
    except:
        try:
            return ox.nearest_nodes(G, lon, lat)
        except:
            nodes, coords = zip(*[(n,(d.get('y'),d.get('x'))) for n,d in G.nodes(data=True)])
            coords_arr = np.array(coords)
            dists = np.hypot(coords_arr[:,0]-lat, coords_arr[:,1]-lon)
            return nodes[int(np.argmin(dists))]

point_nodes_full = [snap_to_node(G_full, p) for p in all_points]

# map full nodes to nearest truck nodes (may map to different ids)
def map_to_truck_node(Gt, Gf, node_full):
    lat = Gf.nodes[node_full]['y']; lon = Gf.nodes[node_full]['x']
    try:
        return ox.get_nearest_node(Gt, (lat, lon))
    except:
        try:
            return ox.nearest_nodes(Gt, lon, lat)
        except:
            nodes, coords = zip(*[(n,(d.get('y'),d.get('x'))) for n,d in Gt.nodes(data=True)])
            coords_arr = np.array(coords)
            dists = np.hypot(coords_arr[:,0]-lat, coords_arr[:,1]-lon)
            return nodes[int(np.argmin(dists))]
point_nodes_truck = [map_to_truck_node(G_truck, G_full, nf) for nf in point_nodes_full]

# path getter: prefer truck graph, fallback to full graph
def get_best_path(n1_truck, n2_truck, n1_full, n2_full):
    try:
        return nx.shortest_path(G_truck, n1_truck, n2_truck, weight='travel_time'), 'truck'
    except (NetworkXNoPath, nx.NodeNotFound):
        pass
    try:
        return nx.shortest_path(G_full, n1_full, n2_full, weight='travel_time'), 'full'
    except (NetworkXNoPath, nx.NodeNotFound):
        return None, None

# travel-time matrix for solver (use full graph distances)
N = len(all_points)
tt_matrix = np.zeros((N,N), dtype=float)
for i, ni in enumerate(point_nodes_full):
    lengths = nx.single_source_dijkstra_path_length(G_full, ni, weight='travel_time')
    for j, nj in enumerate(point_nodes_full):
        tt_matrix[i,j] = lengths.get(nj, np.inf)

# OR-Tools VRP with capacity
manager = pywrapcp.RoutingIndexManager(N, K, [n_bins + (v % n_starts) for v in range(K)], [n_bins + n_starts]*K)
routing = pywrapcp.RoutingModel(manager)

def time_callback(from_index, to_index):
    return int(tt_matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])
transit_cb_idx = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
routing.AddDimension(transit_cb_idx, 0, int(1e9), True, "Time")
time_dim = routing.GetDimensionOrDie("Time")
time_dim.SetGlobalSpanCostCoefficient(1000)

# demands: bins=1, starts+sink=0
demands = [1]*n_bins + [0]*(n_starts+1)
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    return int(demands[node])
demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
capacities = [capacity_nodes]*K
routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, capacities, True, "Capacity")

# search params
search_params = pywrapcp.DefaultRoutingSearchParameters()
search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
search_params.time_limit.seconds = 30

solution = routing.SolveWithParameters(search_params)
if not solution:
    raise SystemExit("No solution found (try increasing time_limit or relaxing constraints).")

# extract routes and assigned bins per truck
routes = []
truck_assignments = {}
for v in range(K):
    idx = routing.Start(v)
    route = []
    while not routing.IsEnd(idx):
        route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    route.append(manager.IndexToNode(idx))
    routes.append(route)
    bins_for_truck = [r for r in route if r < n_bins]
    truck_assignments[f"Truck_{v+1}"] = bins_for_truck

print("\n=== Bin assignments ===")
for t, bl in truck_assignments.items():
    print(f"{t}: {len(bl)} bins -> {bl}")

# ---------------------------
# Build detailed road-following path (list of (lat,lon) per truck)
# Also compute time (seconds) per truck and overall
# ---------------------------
service_time_per_bin = 0  # seconds per stop, change if you want e.g. 3*60 for 3 minutes per pickup

truck_route_coords = []
truck_route_nodes = []   # store full node-id list for each truck route (graph nodes)
truck_times_seconds = [] # travel time (seconds) per truck (including service_time_per_bin)

def haversine_meters(a, b):
    # a, b are (lat, lon)
    R = 6371000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(s), math.sqrt(1-s))

for t_idx, route in enumerate(routes):
    coords = []
    nodes_seq = []
    time_s = 0.0
    for i in range(len(route)-1):
        a = route[i]; b = route[i+1]
        n1_tr = point_nodes_truck[a]; n2_tr = point_nodes_truck[b]
        n1_full = point_nodes_full[a]; n2_full = point_nodes_full[b]
        path, which = get_best_path(n1_tr, n2_tr, n1_full, n2_full)
        if path is None:
            # fallback to straight segment between the two original points (rare)
            coords.append(all_points[a]); coords.append(all_points[b])
            # estimate time using haversine and default speed 40 km/h
            dist_m = haversine_meters(all_points[a], all_points[b])
            time_s += dist_m / (40 * 1000.0 / 3600.0)
            print(f"Warning: no graph path between indices {a} and {b}; drawing straight line and estimating time")
        else:
            G_use = G_truck if which=='truck' else G_full
            path_coords = [(G_use.nodes[n]['y'], G_use.nodes[n]['x']) for n in path]
            if coords and coords[-1] == path_coords[0]:
                coords.extend(path_coords[1:])
            else:
                coords.extend(path_coords)
            # append node ids, avoid duplicating node between segments
            if nodes_seq and nodes_seq[-1] == path[0]:
                nodes_seq.extend(path[1:])
            else:
                nodes_seq.extend(path)
            # compute segment time by summing edge travel_time along path (handle multigraph)
            for u, v in zip(path[:-1], path[1:]):
                edge_data = G_use.get_edge_data(u, v)
                if edge_data is None:
                    edge_data = G_use.get_edge_data(v, u) or {}
                if edge_data:
                    best = None
                    # MultiDiGraph: edge_data is dict keyed by key -> attr dict
                    for key, attr in edge_data.items():
                        tt = attr.get('travel_time')
                        if tt is None:
                            length = attr.get('length', 0)
                            hw = attr.get('highway')
                            if isinstance(hw, list): hw = hw[0]
                            speed_kph = SPEEDS.get(hw, 40)
                            tt = length / (max(speed_kph, 1) * 1000.0/3600.0)
                        if best is None or tt < best:
                            best = tt
                    if best is not None:
                        time_s += best
                else:
                    # fallback estimate by node coordinates
                    latlon_u = (G_use.nodes[u]['y'], G_use.nodes[u]['x'])
                    latlon_v = (G_use.nodes[v]['y'], G_use.nodes[v]['x'])
                    dist_m = haversine_meters(latlon_u, latlon_v)
                    time_s += dist_m / (40 * 1000.0 / 3600.0)
    # add service time
    bins_collected = truck_assignments.get(f"Truck_{t_idx+1}", [])
    time_s += service_time_per_bin * len(bins_collected)

    truck_route_coords.append(coords)
    truck_route_nodes.append(nodes_seq)
    truck_times_seconds.append(time_s)

# print per-truck and overall times
total_seconds = sum(truck_times_seconds)
print("\n=== Travel times ===")
for i, tsec in enumerate(truck_times_seconds):
    mins = tsec / 60.0
    print(f"Truck {i+1}: {mins:.1f} min ({tsec:.0f} s) — {len(truck_assignments.get(f'Truck_{i+1}', []))} bins")
print(f"Overall total travel time (sum of trucks): {total_seconds/60.0:.1f} min ({total_seconds/3600.0:.2f} h)")
print(f"Makespan (longest single truck): {max(truck_times_seconds)/60.0:.1f} min")

# ---------- Plot combined map ----------
fig, ax = ox.plot_graph(G_full, show=False, close=False, node_size=0, edge_color="lightgrey", bgcolor="white", figsize=(10,10))
colors = ['red','blue','green','orange','purple','cyan','magenta']

# plot each truck route (road-following)
for i, coords in enumerate(truck_route_coords):
    if not coords: continue
    lats, lons = zip(*coords)
    ax.plot(lons, lats, color=colors[i%len(colors)], linewidth=3, alpha=0.9, label=f"Truck {i+1} ({truck_times_seconds[i]/60.0:.1f} min)")

# plot bins colored by assigned truck
for t_idx, (truck, bin_list) in enumerate(truck_assignments.items()):
    for b_idx in bin_list:
        lat, lon = bins[b_idx]
        ax.scatter(lon, lat, c=colors[t_idx%len(colors)], s=60, zorder=6)
        ax.text(lon, lat, f"B{b_idx}", fontsize=8, color=colors[t_idx%len(colors)], ha='right', va='bottom')

# plot unassigned bins (shouldn't happen because of capacity check)
assigned = set(sum(list(truck_assignments.values()), []))
for i in range(n_bins):
    if i not in assigned:
        lat, lon = bins[i]
        ax.scatter(lon, lat, c='black', s=40, zorder=6)
        ax.text(lon, lat, f"B{i}", fontsize=8, color='black', ha='right', va='bottom')

# plot starts and sink
for i,(lat,lon) in enumerate(starts_4):
    ax.scatter(lon, lat, c='black', marker='s', s=90, zorder=7)
    ax.text(lon, lat, f"S{i}", fontsize=9, color='black', ha='left', va='bottom')
ax.scatter(sink[1], sink[0], c='gold', s=200, marker='*', edgecolor='black', zorder=8)
ax.text(sink[1], sink[0], "Sink", fontsize=10, color='darkorange', ha='left', va='bottom')

ax.legend(); plt.title("All Trucks - Routes and Assigned Bins")
plt.show()

# ---------- Plot separate maps per driver ----------
for i, coords in enumerate(truck_route_coords):
    fig, ax = ox.plot_graph(G_full, show=False, close=False, node_size=0, edge_color="lightgrey", bgcolor="white", figsize=(8,8))
    if coords:
        lats, lons = zip(*coords)
        ax.plot(lons, lats, color=colors[i%len(colors)], linewidth=3, alpha=0.95, label=f"Truck {i+1} ({truck_times_seconds[i]/60.0:.1f} min)")
    # plot bins assigned to this truck
    bin_list = truck_assignments[f"Truck_{i+1}"]
    for b_idx in bin_list:
        lat, lon = bins[b_idx]
        ax.scatter(lon, lat, c=colors[i%len(colors)], s=90, zorder=6)
        ax.text(lon, lat, f"B{b_idx}", fontsize=9, color=colors[i%len(colors)], ha='right', va='bottom')
    # plot start for this truck (where it started)
    start_idx = (n_bins + (i % n_starts))
    start_latlon = all_points[start_idx]
    ax.scatter(start_latlon[1], start_latlon[0], c='black', marker='s', s=120, zorder=7)
    ax.text(start_latlon[1], start_latlon[0], f"S{start_idx-n_bins}", fontsize=9, color='black', ha='left', va='bottom')
    # sink
    ax.scatter(sink[1], sink[0], c='gold', s=150, marker='*', edgecolor='black', zorder=8)
    ax.text(sink[1], sink[0], "Sink", fontsize=9, color='darkorange', ha='left', va='bottom')
    ax.legend(); plt.title(f"Truck {i+1} - Route and Assigned Bins ({truck_times_seconds[i]/60.0:.1f} min)")
    plt.show()