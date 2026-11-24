# Required imports
import numpy as np  # Numerical computing library for array operations
import matplotlib.pyplot as plt  # Plotting library for visualization
import sys  # System-specific parameters and functions for command line arguments
import json  # JSON encoder and decoder for scenario configuration
import os  # Operating system interfaces for file path handling
from Map import Map  # Custom Map class for radar and detection map handling
from Boundaries import Boundaries  # Custom class for geographic boundaries
from SearchEngine import build_graph, path_finding, compute_path_cost, h1, make_h2, make_h3  # Pathfinding functions
from matplotlib.colors import ListedColormap  # Custom colormap generation for visualization

# Scenario file, resolved next to this script so the program can be run from any directory
SCENARIOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")

# Seed for the pseudo-random radar generation (fixed by the assignment, DO NOT MODIFY)
RANDOM_SEED = 42


def plot_all_results(boundaries: Boundaries,
                     radar_locations: np.array,
                     detection_map: np.array,
                     solution_plan: list,
                     tolerance: float,
                     pois: np.array,
                     title: str = "") -> None:
    """
    Visualizes radar locations, detection map, navigable areas, and flight path in a 4-panel plot.

    Parameters:
        boundaries (Boundaries): Geographic boundaries of the map area
        radar_locations (np.array): Array of radar coordinates (latitude, longitude)
        detection_map (np.array): 2D array of detection probabilities for each grid cell
        solution_plan (list): List of paths (lists of (row, col) tuples) connecting POIs
        tolerance (float): Maximum allowed detection probability for navigation
        pois (np.array): Array of Points Of Interest coordinates
        title (str): Figure title, used to tell apart the figure of each heuristic

    Returns:
        None: Displays matplotlib figure with visualization
    """
    # Create a large figure for visualization (22 inches wide, 6 inches tall)
    plt.figure(figsize=(22, 6), facecolor='white')
    if title:
        plt.suptitle(title)

    # Define a color cycle for path segments using matplotlib's tab10 colormap
    path_colors = plt.cm.tab10.colors

    # Style configuration dictionary with smaller markers and consistent styling
    style = {
        'start': {'marker': 'P', 's': 100, 'edgecolor': 'black', 'zorder': 5},  # Start point style
        'end': {'marker': 'X', 's': 100, 'edgecolor': 'black', 'zorder': 5},  # End point style
        'poi': {'marker': 'o', 's': 50, 'edgecolor': 'black', 'zorder': 4},  # POI style
        'radar': {'s': 20, 'edgecolor': 'white', 'zorder': 3},  # Radar style
        'path': {'linewidth': 2, 'zorder': 2},  # Path line style
        'boundary': {'color': 'black', 'linestyle': '--', 'linewidth': 1.2, 'zorder': 1}  # Boundary style
    }

    def to_grid(lat: float, lon: float) -> tuple:
        """
        Converts geographic coordinates (latitude, longitude) to grid indices (i,j).

        Parameters:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate

        Returns:
            tuple: (y, x) grid indices corresponding to the input coordinates
        """
        # Normalize longitude to grid x-coordinate
        x = int((lon - boundaries.min_lon) / (boundaries.max_lon - boundaries.min_lon) * (detection_map.shape[1] - 1))
        # Normalize latitude to grid y-coordinate
        y = int((lat - boundaries.min_lat) / (boundaries.max_lat - boundaries.min_lat) * (detection_map.shape[0] - 1))
        return (y, x)

    # Define the four subplots to be displayed
    plots = [
        {'title': "Radar Locations", 'type': 'geo', 'data': None},  # Geographic radar view
        {'title': "Detection Heatmap", 'type': 'heat', 'data': detection_map},  # Detection probability
        {'title': "Navigable Areas", 'type': 'binary', 'data': (detection_map <= tolerance)},  # Binary navigability
        {'title': "Flight Path", 'type': 'solution', 'data': solution_plan}  # Optimal path visualization
    ]

    # Create each subplot in a 1x4 grid
    for i, plot in enumerate(plots, 1):
        plt.subplot(1, 4, i)
        plt.title(plot['title'], pad=15, y=1.02)  # Add title with padding

        # Plot boundaries first (zorder=1 ensures they stay in background)
        if plot['type'] == 'geo':
            # Plot geographic boundaries as a rectangle
            plt.plot(
                [boundaries.min_lon, boundaries.max_lon, boundaries.max_lon, boundaries.min_lon, boundaries.min_lon],
                [boundaries.min_lat, boundaries.min_lat, boundaries.max_lat, boundaries.max_lat, boundaries.min_lat],
                **style['boundary']
            )
        elif plot['type'] in ('heat', 'binary', 'solution'):
            # Plot grid boundaries as a rectangle
            plt.plot(
                [0, detection_map.shape[1] - 1, detection_map.shape[1] - 1, 0, 0],
                [0, 0, detection_map.shape[0] - 1, detection_map.shape[0] - 1, 0],
                **style['boundary']
            )
            # Add padding to prevent clipping of elements
            plt.xlim(-0.5, detection_map.shape[1] - 0.5)
            plt.ylim(-0.5, detection_map.shape[0] - 0.5)

        # Plot specific content for each subplot type
        if plot['type'] == 'geo':
            # Plot radar locations and POIs on geographic map
            plt.scatter(radar_locations[:, 1], radar_locations[:, 0], color='red', **style['radar'])
            plt.scatter(pois[0][1], pois[0][0], color='lime', **style['start'])  # Start POI
            plt.scatter(pois[-1][1], pois[-1][0], color='cyan', **style['end'])  # End POI
            if len(pois) > 2:
                plt.scatter(pois[1:-1, 1], pois[1:-1, 0], color='orange', **style['poi'])  # Intermediate POIs

        elif plot['type'] in ('heat', 'binary'):
            # Plot detection heatmap or binary navigability map
            cmap = 'YlGnBu' if plot['type'] == 'heat' else ListedColormap(['white', 'limegreen'])
            im = plt.imshow(plot['data'], cmap=cmap, origin='lower')
            plt.colorbar(im, pad=0.02, label='Detection' if plot['type'] == 'heat' else 'Navigable\n(0=No, 1=Yes)')

            # Plot POIs on grid coordinates
            start_yx = to_grid(pois[0][0], pois[0][1])
            end_yx = to_grid(pois[-1][0], pois[-1][1])
            plt.scatter(start_yx[1], start_yx[0], color='lime', **style['start'])
            plt.scatter(end_yx[1], end_yx[0], color='cyan', **style['end'])
            if len(pois) > 2:
                for lat, lon in pois[1:-1]:
                    y, x = to_grid(lat, lon)
                    plt.scatter(x, y, color='orange', **style['poi'])

        # Solution path visualization with different colors for each segment
        if plot['type'] == 'solution':
            if solution_plan:
                # Display detection map as semi-transparent background
                plt.imshow(detection_map, cmap='YlGnBu', origin='lower', alpha=0.3)
                # Plot each segment of the solution path with different colors
                for idx, path in enumerate(solution_plan):
                    color = path_colors[idx % len(path_colors)]  # Cycle through colors
                    points = np.array(path)  # (row, col) tuples to a numpy array
                    plt.plot(points[:, 1], points[:, 0], color=color, **style['path'])
                    # Use same color for start/end markers of each segment
                    plt.scatter(points[0, 1], points[0, 0], color=color,
                                **{k: v for k, v in style['start'].items() if k != 'color'})
                    plt.scatter(points[-1, 1], points[-1, 0], color=color,
                                **{k: v for k, v in style['end'].items() if k != 'color'})
            else:
                # No path found message
                plt.text(detection_map.shape[1] / 2, detection_map.shape[0] / 2,
                         'NO SAFE PATH EXISTS\n(Tolerance: {:.2f})'.format(tolerance),
                         ha='center', va='center', fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.8))
                plt.imshow((detection_map <= tolerance), cmap=ListedColormap(['white', 'limegreen']), origin='lower')

        # Configure axis labels and grid
        plt.xlabel("Longitude" if plot['type'] == 'geo' else "Grid X", labelpad=8)
        plt.ylabel("Latitude" if plot['type'] == 'geo' else "Grid Y", labelpad=8)
        plt.grid(True, alpha=0.3) if plot['type'] == 'geo' else plt.grid(False)

    # Create legend with smaller markers
    handles = [
        plt.Line2D([0], [0], marker=style['start']['marker'], color='w', label='Start',
                   markerfacecolor='lime', markersize=8),
        plt.Line2D([0], [0], marker=style['end']['marker'], color='w', label='End',
                   markerfacecolor='cyan', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Radars',
                   markerfacecolor='red', markersize=6),
        plt.Line2D([0], [0], marker='o', color='w', label='POIs',
                   markerfacecolor='orange', markersize=6)
    ]
    plt.figlegend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))

    # Adjust layout to prevent overlap and display the figure
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()


def parse_args() -> dict:
    """
    Parses command line arguments and loads scenario configuration from JSON file.

    Returns:
        dict: Dictionary containing execution parameters including:
              - Geographic boundaries
              - Grid dimensions (width, height)
              - Number of radars
              - Points of Interest (POIs)
              - Detection tolerance

    Raises:
        SystemExit: If the arguments are missing or invalid, the JSON file is not found,
                    or the scenario does not exist
    """
    if len(sys.argv) != 3:
        print("Usage: python main.py <scenario_name> <tolerance>")
        print("Example: python main.py scenario_8 0.75")
        sys.exit(1)

    # Get scenario name and tolerance from command line arguments
    scenario_json = sys.argv[1]  # First argument: scenario name
    try:
        tolerance = float(sys.argv[2])  # Second argument: tolerance value
    except ValueError:
        print(f"Error: tolerance must be a number, got '{sys.argv[2]}'")
        sys.exit(1)

    execution_parameters = {}

    # Load scenario configuration from JSON file
    try:
        with open(SCENARIOS_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for entry in data:
                key = list(entry.keys())[0]
                if key == scenario_json:
                    execution_parameters = entry[key]
                    break
    except FileNotFoundError:
        print(f"Error: Scenario file not found at {SCENARIOS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in scenarios file")
        sys.exit(1)

    if not execution_parameters:
        print(f"Error: scenario '{scenario_json}' not found in {SCENARIOS_PATH}")
        sys.exit(1)

    # Add tolerance to execution parameters
    execution_parameters["tolerance"] = tolerance
    return execution_parameters


def main() -> None:
    """
    Main execution function for the path planning system.
    Orchestrates the entire process from loading scenario to visualization.
    """
    # Parse command line arguments to get scenario configuration
    execution_parameters = parse_args()

    # Set the pseudo-random number generator seed (DO NOT MODIFY)
    np.random.seed(RANDOM_SEED)

    # Initialize geographic boundaries from scenario parameters
    boundaries = Boundaries(
        max_lat=execution_parameters['max_lat'],
        min_lat=execution_parameters['min_lat'],
        max_lon=execution_parameters['max_lon'],
        min_lon=execution_parameters['min_lon']
    )

    # Create map instance with specified dimensions
    M = Map(
        boundaries=boundaries,
        height=execution_parameters['H'],
        width=execution_parameters['W']
    )

    # Generate random radars with properties based on scenario
    M.generate_radars(n_radars=execution_parameters['n_radars'])
    radar_locations = M.get_radars_locations_numpy()

    # Compute detection probability map across the entire grid
    detection_map = M.compute_detection_map()

    # Build navigation graph (only cells below tolerance are traversable)
    tolerance = execution_parameters['tolerance']
    G = build_graph(detection_map=detection_map, tolerance=tolerance)

    # Get Points of Interest from scenario
    POIs = np.array(execution_parameters['POIs'], dtype=np.float32)

    # Heuristics to compare (h2 and h3 depend on the map, so they are built from it)
    heuristics = {
        "h1 (Manhattan x EPSILON)": h1,
        "h2 (Manhattan x global minimum detection)": make_h2(detection_map),
        "h3 (adaptive risk-aware, not admissible)": make_h3(detection_map, tolerance),
    }

    # Run A* with each heuristic and display results
    solution_plans = {}
    for name, heuristic in heuristics.items():
        print(f"\nTesting {name}:")
        solution_plan, nodes_generated = path_finding(
            G=G,
            heuristic_function=heuristic,
            locations=POIs,
            initial_location_index=0,
            boundaries=boundaries,
            map_width=M.width,
            map_height=M.height
        )
        if solution_plan:
            print(f"Path cost: {compute_path_cost(G, solution_plan)}")
        else:
            print("No safe path exists for this tolerance")
        print(f"Nodes generated: {nodes_generated}")
        solution_plans[name] = solution_plan

    # Visualize results for each heuristic
    try:
        for name, solution_plan in solution_plans.items():
            plot_all_results(
                boundaries,
                radar_locations,
                detection_map,
                solution_plan,
                tolerance,
                POIs,
                title=name
            )
    except ValueError as e:
        print(f"Visualization Error: {e}")
        # Fallback visualization if solution fails
        plt.figure(figsize=(12, 6))

        # Radar locations plot
        plt.subplot(1, 2, 1)
        plt.title("Radar Locations")
        plt.plot(
            [boundaries.min_lon, boundaries.max_lon, boundaries.max_lon, boundaries.min_lon, boundaries.min_lon],
            [boundaries.max_lat, boundaries.max_lat, boundaries.min_lat, boundaries.min_lat, boundaries.max_lat],
            label='Boundaries', linestyle='--', c='black', zorder=1
        )
        plt.scatter(radar_locations[:, 1], radar_locations[:, 0], label='Radars', c='red', zorder=2)
        plt.legend()

        # Detection field plot
        plt.subplot(1, 2, 2)
        plt.title("Detection Fields")
        plt.imshow(detection_map, cmap='Greens', origin='lower')
        plt.colorbar(label='Detection Probability')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Entry point when script is executed directly
    main()