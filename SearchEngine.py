# Required imports
import numpy as np  # For numerical operations and array handling
import networkx as nx  # For graph operations and pathfinding
from Boundaries import Boundaries  # Custom class for geographic boundaries
from Map import EPSILON  # Minimum detection value constant (1e-4)

# Allowed moves on the grid: up, down, left, right (no diagonal movement)
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

# h3 risk threshold: theta = min(H3_THETA_CAP, H3_THETA_TOLERANCE_FACTOR * tolerance)
H3_THETA_CAP = 0.5
H3_THETA_TOLERANCE_FACTOR = 0.7


def manhattan_distance(current_node: tuple, goal_node: tuple) -> int:
    """
    Number of 4-directional moves between two cells if there were no obstacles.

    Args:
        current_node: Tuple (i,j) representing current grid position
        goal_node: Tuple (i,j) representing target grid position

    Returns:
        int: |i_goal - i| + |j_goal - j|
    """
    return abs(goal_node[0] - current_node[0]) + abs(goal_node[1] - current_node[1])


def h1(current_node: tuple, goal_node: tuple) -> float:
    """
    Manhattan distance scaled by the minimum cost of a move (EPSILON).

    Every move costs at least EPSILON (the detection map is scaled to [EPSILON, 1]),
    and at least manhattan_distance moves are needed to reach the goal, so:
        - Admissible: never overestimates the real cost
        - Consistent: moving one cell changes h by at most EPSILON, which is
          never more than the cost of that move

    Args:
        current_node: Tuple (i,j) representing current grid position
        goal_node: Tuple (i,j) representing target grid position

    Returns:
        float: Estimated path cost from current to goal node
    """
    return EPSILON * manhattan_distance(current_node, goal_node)


def make_h2(detection_map: np.array):
    """
    Builds h2: Manhattan distance scaled by the global minimum detection value.

    The minimum is computed once here, not on every heuristic call (on a
    1024x1024 map that would mean scanning a million cells per call).

    Admissible and consistent for the same reason as h1. Note that, because
    Map.compute_detection_map scales every map to [EPSILON, 1], the global
    minimum is always EPSILON, so h2 returns exactly the same values as h1.

    Args:
        detection_map: 2D array of detection probabilities (already scaled)

    Returns:
        Callable (current_node, goal_node) -> float
    """
    min_detection = float(np.min(detection_map))

    def h2(current_node: tuple, goal_node: tuple) -> float:
        return min_detection * manhattan_distance(current_node, goal_node)

    return h2


def make_h3(detection_map: np.array, tolerance: float):
    """
    Builds h3: adaptive risk-aware heuristic (exploratory extension).

    If the current cell's detection is below the threshold
    theta = min(H3_THETA_CAP, H3_THETA_TOLERANCE_FACTOR * tolerance),
    the Manhattan distance is scaled by that local detection value.
    Otherwise it falls back to the global minimum detection value.

    Properties:
        - NOT admissible: when the cells between the current one and the goal
          are cheaper than the current cell, the estimate is too high. In
          exchange, A* explores far fewer nodes, and the paths it returns can be
          slightly more expensive than the optimal ones (same idea as weighted A*).

    Args:
        detection_map: 2D array of detection probabilities (already scaled)
        tolerance: Maximum detection probability allowed for navigation

    Returns:
        Callable (current_node, goal_node) -> float
    """
    min_detection = float(np.min(detection_map))
    theta = min(H3_THETA_CAP, H3_THETA_TOLERANCE_FACTOR * tolerance)

    def h3(current_node: tuple, goal_node: tuple) -> float:
        local_detection = float(detection_map[current_node])
        scaling_factor = local_detection if local_detection <= theta else min_detection
        return scaling_factor * manhattan_distance(current_node, goal_node)

    return h3


def build_graph(detection_map: np.array, tolerance: np.float32) -> nx.DiGraph:
    """
    Constructs a directed graph from the detection map where:
    - Nodes represent grid cells
    - Edges represent valid transitions between adjacent cells
    - Edge weights equal the detection probability of the destination cell

    Args:
        detection_map: 2D array of detection probabilities (0-1)
        tolerance: Maximum allowed detection probability for navigation

    Returns:
        nx.DiGraph: Directed graph representing traversable paths

    Note:
        Only creates edges between cells where both source and destination
        have detection probabilities <= tolerance
    """
    G = nx.DiGraph()
    height, width = detection_map.shape

    # Add all grid cells as nodes with their detection values
    for i in range(height):
        for j in range(width):
            G.add_node((i, j), detection=detection_map[i, j])

    # Create edges between navigable adjacent cells (4-directional)
    for i in range(height):
        for j in range(width):
            if detection_map[i, j] > tolerance:
                continue  # Skip non-navigable cells

            for di, dj in MOVES:
                ni, nj = i + di, j + dj
                # Verify neighbor is within bounds and navigable
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_detection = detection_map[ni, nj]
                    if neighbor_detection <= tolerance:
                        # Edge weight is the detection of the destination cell
                        G.add_edge((i, j), (ni, nj), weight=neighbor_detection)

    return G


def discretize_coords(high_level_plan: np.array, boundaries: Boundaries,
                      map_width: np.int32, map_height: np.int32) -> list:
    """
    Converts geographic coordinates (lat,lon) to grid indices (i,j).

    Args:
        high_level_plan: Array of (lat,lon) coordinates for POIs
        boundaries: Geographic boundaries of the map
        map_width: Number of columns in the grid
        map_height: Number of rows in the grid

    Returns:
        list: (row,col) grid indices as tuples of ints, ready to be used as graph nodes

    Process:
        1. Normalizes coordinates to [0,1] range relative to boundaries
        2. Scales to grid dimensions
        3. Returns integer indices using (row,column) convention
    """
    discretized = []
    for lat, lon in high_level_plan:
        # Normalize longitude and latitude to [0,1] range
        norm_x = (lon - boundaries.min_lon) / (boundaries.max_lon - boundaries.min_lon)
        norm_y = (lat - boundaries.min_lat) / (boundaries.max_lat - boundaries.min_lat)

        # Convert to grid indices (0 to width-1 / height-1)
        col = int(norm_x * (map_width - 1))
        row = int(norm_y * (map_height - 1))
        discretized.append((row, col))  # Matrix-style (row,col) convention

    return discretized


def path_finding(G: nx.DiGraph, heuristic_function, locations: np.array,
                 initial_location_index: np.int32, boundaries: Boundaries,
                 map_width: np.int32, map_height: np.int32) -> tuple:
    """
    Plans paths through consecutive POIs using A*.

    Args:
        G: Navigation graph from build_graph()
        heuristic_function: Callable (current_node, goal_node) -> float,
                            i.e. h1 or the result of make_h2 / make_h3
        locations: Array of (lat,lon) POI coordinates
        initial_location_index: Starting POI index
        boundaries: Geographic boundaries
        map_width: Grid width
        map_height: Grid height

    Returns:
        tuple: (solution_plan, nodes_generated) where:
            - solution_plan: List of paths (lists of (row,col) tuples) between consecutive POIs,
                             or an empty list if any segment is unreachable
            - nodes_generated: Number of heuristic evaluations. NetworkX's A*
                               evaluates the heuristic once per newly discovered
                               node, so this counts generated nodes
    """
    nodes_generated = 0

    def counted_heuristic(current_node, goal_node):
        nonlocal nodes_generated
        nodes_generated += 1
        return heuristic_function(current_node, goal_node)

    # Convert POIs from geographic to grid coordinates
    discrete_locations = discretize_coords(locations, boundaries, map_width, map_height)

    solution_plan = []

    # Find path between each consecutive POI pair
    for current_index in range(initial_location_index, len(discrete_locations) - 1):
        start = discrete_locations[current_index]
        goal = discrete_locations[current_index + 1]

        try:
            path = nx.astar_path(G, start, goal, heuristic=counted_heuristic, weight='weight')
        except nx.NetworkXNoPath:
            print(f"No path found from {start} to {goal}")
            return [], nodes_generated

        solution_plan.append(path)

    return solution_plan, nodes_generated


def compute_path_cost(G: nx.DiGraph, solution_plan: list) -> float:
    """
    Calculates total detection exposure cost for a solution plan.

    Args:
        G: Navigation graph with edge weights
        solution_plan: List of path segments from path_finding()

    Returns:
        float: Sum of all edge weights along the plan

    Note:
        Each edge weight represents the detection probability at the destination cell.
        Lower values indicate safer (less detectable) paths.
    """
    total_cost = 0.0

    for path in solution_plan:
        for u, v in zip(path, path[1:]):
            total_cost += float(G.edges[u, v]['weight'])

    return total_cost