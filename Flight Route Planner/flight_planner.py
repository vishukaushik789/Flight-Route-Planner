import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import urllib.request                                #to fetch image directly from url 
import io                                            #to fetch image directly from url 
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import heapq                                          #used for best-first/Dijkstra Algot 
from collections import deque                         #used for breadth first search

# -----------------------------
# Step 1: Define airports (nodes) and coordinates
# -----------------------------
airport_coords = {
    'DEL': (0, 0),     # Delhi
    'MUM': (4, -1),    # Mumbai
    'BLR': (6, -3),    # Bengaluru
    'HYD': (5, -2),    # Hyderabad
    'CCU': (7, 2),     # Kolkata
    'DXB': (10, -1),   # Dubai
    'SIN': (12, -3),   # Singapore
    'LHR': (15, 3)     # London
}

# -----------------------------
# Step 2: Define connections (edges)
# -----------------------------
#---distance in km-----
edges = [
    ('DEL', 'MUM', 11),
    ('DEL', 'HYD', 10),
    ('DEL', 'CCU', 13),
    ('MUM', 'BLR', 8),
    ('HYD', 'BLR', 3),
    ('HYD', 'CCU', 10),
    ('BLR', 'SIN', 9),
    ('MUM', 'DXB', 6),
    ('DXB', 'LHR', 7),
    ('CCU', 'SIN', 10),
    ('SIN', 'LHR', 15),
    ('HYD', 'DXB', 8)
]

# Create the graph
G = nx.Graph()
for u, v, dist in edges:
    weather_penalty = random.randint(0, 2)
    congestion = random.uniform(0.9, 1.3)
    total_cost = dist * congestion + weather_penalty
    G.add_edge(u, v, weight=total_cost, distance=dist, penalty=weather_penalty)

# -----------------------------
# Algorithm 1: Dijkstra's Algorithm
# -----------------------------
def dijkstra_flight_planner(graph, start, goal):
    """Shortest path using Dijkstra's algorithm"""
    pq = [(0, start, [start])]
    visited = set()
    
    while pq:
        cost, current, path = heapq.heappop(pq)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current == goal:
            return path, cost
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                edge_cost = graph[current][neighbor]['weight']
                new_cost = cost + edge_cost
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, neighbor, new_path))
    
    return None, float('inf')

# -----------------------------
# Algorithm 2: BFS (Breadth-First Search)
# -----------------------------
def bfs_flight_planner(graph, start, goal):
    """Find path with minimum stops using BFS"""
    queue = deque([(start, [start], 0)])
    visited = {start}
    
    while queue:
        current, path, cost = queue.popleft()
        
        if current == goal:
            return path, cost
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                edge_cost = graph[current][neighbor]['weight']
                new_cost = cost + edge_cost
                queue.append((neighbor, path + [neighbor], new_cost))
    
    return None, float('inf')

# -----------------------------
# Algorithm 3: DFS (Depth-First Search)
# -----------------------------
def dfs_flight_planner(graph, start, goal):
    """Find path using DFS"""
    stack = [(start, [start], 0)]
    visited = set()
    best_path = None
    best_cost = float('inf')
    
    while stack:
        current, path, cost = stack.pop()
        
        if current == goal:
            if cost < best_cost:
                best_path = path
                best_cost = cost
            continue
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                edge_cost = graph[current][neighbor]['weight']
                stack.append((neighbor, path + [neighbor], cost + edge_cost))
    
    return best_path, best_cost if best_path else (None, float('inf'))

# -----------------------------
# Algorithm 4: A* Algorithm (Heuristic-based)
# -----------------------------
def heuristic(a, b):
    """Euclidean distance heuristic"""
    (x1, y1), (x2, y2) = airport_coords[a], airport_coords[b]
    return math.sqrt((x1 - x2)*2 + (y1 - y2)*2)

def astar_flight_planner(graph, start, goal):
    """A* algorithm with heuristic"""
    open_set = {start}
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes}
    f_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0
    f_score[start] = heuristic(start, goal)

    while open_set:
        current = min(open_set, key=lambda x: f_score[x])
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        open_set.remove(current)
        for neighbor in graph.neighbors(current):
            edge_data = graph[current][neighbor]
            travel_cost = edge_data['weight']
            tentative_g = g_score[current] + travel_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                open_set.add(neighbor)

    return None, float('inf')

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("✈️ Flight Route Planner - Multiple Algorithms")
root.geometry("500x550")

# Global variable to store background image reference
bg_image = None

# Load background image from URL
try:
    image_url = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=800"
    with urllib.request.urlopen(image_url) as url:
        image_data = url.read()
    
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((500, 550), Image.Resampling.LANCZOS)
    bg_image = ImageTk.PhotoImage(image)
    
    # Create a canvas for background
    canvas = tk.Canvas(root, width=500, height=550)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")
    
except Exception as e:
    print(f"Could not load background image: {e}")
    canvas = tk.Canvas(root, width=500, height=550, bg='skyblue')
    canvas.pack(fill="both", expand=True)

def find_route():
    """Find route based on selected algorithm"""
    start = start_var.get()
    end = end_var.get()
    algo = algo_var.get()

    if start == end:
        messagebox.showwarning("Invalid Input", "Start aur destination airport alag hone chahiye!")
        return

    # Algorithm selection dictionary
    algorithms = {
        "Dijkstra": dijkstra_flight_planner,
        "BFS (Breadth-First Search)": bfs_flight_planner,
        "DFS (Depth-First Search)": dfs_flight_planner,
        "A* (Heuristic)": astar_flight_planner
    }
    
    # Select and run the algorithm
    planner = algorithms[algo]
    path, total_cost = planner(G, start, end)
    
    if not path:
        messagebox.showerror("Error", "Koi route nahi mila!")
        return

    # Display results
    result_text = f"Algorithm Used: {algo}\n\n" \
                  f"Route: {' → '.join(path)}\n\n" \
                  f"Total Cost: {total_cost:.2f}\n" \
                  f"Number of Stops: {len(path) - 1}"
    result_label.config(text=result_text)

    # --- Visualize route ---
    pos = airport_coords
    plt.figure(figsize=(12, 8))
    
    # Draw all nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2500, 
            font_size=11, font_weight='bold', edge_color='gray')
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, 
                                 edge_labels={(u, v): f"{d['distance']}km" 
                                            for u, v, d in G.edges(data=True)},
                                 font_size=9)

    # Highlight the optimal path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen', node_size=2500)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color='red', 
                          arrows=True, arrowsize=20, arrowstyle='->')

    plt.title(f"Algorithm: {algo}\nRoute: {start} → {end} | Total Cost = {total_cost:.2f}", 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# -----------------------------
# Create GUI Widgets
# -----------------------------
airports = list(airport_coords.keys())
start_var = tk.StringVar(value=airports[0])
end_var = tk.StringVar(value=airports[-1])
algo_var = tk.StringVar(value="Dijkstra")

# Title
canvas.create_text(250, 30, text="✈️ Flight Route Planner ✈️", 
                  font=("Arial", 16, "bold"), fill="white")

# Algorithm Selection
canvas.create_text(250, 70, text="Select Algorithm:", 
                  font=("Arial", 13, "bold"), fill="white")

algo_menu = ttk.Combobox(root, textvariable=algo_var, 
                         values=["Dijkstra", 
                                "BFS (Breadth-First Search)", 
                                "DFS (Depth-First Search)", 
                                "A* (Heuristic)"],
                         state="readonly", width=25, font=("Arial", 11))
canvas.create_window(250, 100, window=algo_menu)

# Start Airport Selection
canvas.create_text(250, 140, text="Select Start Airport:", 
                  font=("Arial", 13, "bold"), fill="white")
start_menu = tk.OptionMenu(root, start_var, *airports)
start_menu.config(font=("Arial", 11), bg="white", width=15)
canvas.create_window(250, 170, window=start_menu)

# Destination Airport Selection
canvas.create_text(250, 210, text="Select Destination Airport:", 
                  font=("Arial", 13, "bold"), fill="white")
end_menu = tk.OptionMenu(root, end_var, *airports)
end_menu.config(font=("Arial", 11), bg="white", width=15)
canvas.create_window(250, 240, window=end_menu)

# Find Route Button
find_button = tk.Button(root, text="🔍 Find Best Route", command=find_route, 
                       bg="#4CAF50", fg="white", font=("Arial", 13, "bold"), 
                       relief="raised", bd=4, padx=20, pady=5,
                       activebackground="#45a049")
canvas.create_window(250, 290, window=find_button)

# Result Label
result_label = tk.Label(root, text="Algorithm select karein aur route find karein!", 
                       font=("Arial", 11), justify="left", 
                       wraplength=450, bg="white", relief="solid", bd=3, 
                       padx=15, pady=15)
canvas.create_window(250, 400, window=result_label)

# Information Footer
canvas.create_text(250, 520, 
                  text="💡 Tip: Different algorithms give different results!", 
                  font=("Arial", 10, "italic"), fill="yellow")

root.mainloop()