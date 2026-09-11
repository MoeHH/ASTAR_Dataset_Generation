import cupy as cp
import heapq
import random
import matplotlib.pyplot as plt
from config import dataset_config as config

# --- Color and Grid Constants ---
WHITE, BLACK, RED, BLUE, GREEN, YELLOW, GRAY = range(7)

_cmap = None

def _get_cmap():
    """Lazy-initialise the color map on first use."""
    global _cmap
    if _cmap is None:
        _cmap = cp.array([
            [1, 1, 1],        # 0 - White  - Clear Cell
            [0, 0, 0],        # 1 - Black  - Obstacle
            [1, 0, 0],        # 2 - Red    - Visited
            [0, 0, 1],        # 3 - Blue   - On List
            [0, 1, 0],        # 4 - Green  - Start
            [1, 1, 0],        # 5 - Yellow - Goal
            [0.5, 0.5, 0.5],  # 6 - Gray   - Route Path
        ])
    return _cmap

_VIZ_ENABLED = (
    config.get("show_animation", False) or config.get("show_finalimage", False)
)

grid_size = config["grid_size"]
map_grid = cp.zeros(grid_size, dtype=int)

# --- Grid Utility Functions ---
def sub2ind(row, col, shape):
    """Convert 2D grid coordinates to a linear index."""
    return row * shape[1] + col

def ind2sub(index, shape):
    """Convert a linear index to 2D grid coordinates."""
    return index // shape[1], index % shape[1]

def reset_grid():
    """Reset the grid to all white cells. No-op when visualization is disabled."""
    if not _VIZ_ENABLED:
        return
    global map_grid
    map_grid[:] = WHITE
    map_grid[map_grid == GREEN] = WHITE
    map_grid[map_grid == YELLOW] = WHITE

def set_start_and_goal(start_coords, dest_coords):
    """Set the start (green) and goal (yellow) cells. No-op when visualization is disabled."""
    if not _VIZ_ENABLED:
        return
    global map_grid
    map_grid[map_grid == GREEN] = WHITE
    map_grid[map_grid == YELLOW] = WHITE
    map_grid[start_coords[0], start_coords[1]] = GREEN
    map_grid[dest_coords[0], dest_coords[1]] = YELLOW

def set_obstacles(obstacles):
    """
    Set obstacles deterministically (templates mode). No-op when visualization is disabled.
    Args:
        obstacles: iterable of (row, col) 0-based coordinates.
    """
    if not _VIZ_ENABLED:
        return
    global map_grid
    map_grid[map_grid == BLACK] = WHITE
    for r, c in obstacles:
        map_grid[r, c] = BLACK

def generate_random_obstacles(num_obstacles):
    """Randomly place a specified number of obstacles (black cells) on the grid."""
    global map_grid
    map_grid[map_grid == BLACK] = WHITE
    empty_cells = [(r, c) for r in range(grid_size[0]) for c in range(grid_size[1])
                   if map_grid[r, c] == WHITE]
    for _ in range(min(num_obstacles, len(empty_cells))):
        obstacle = random.choice(empty_cells)
        empty_cells.remove(obstacle)
        map_grid[obstacle[0], obstacle[1]] = BLACK

def is_accessible(node, grid, grid_size):
    """Check if a node has at least one non-obstacle neighbor."""
    r, c = node
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
            if grid[nr, nc] != BLACK:
                return True
    return False

# --- Visualization Functions ---
def draw_map(cell_values=None):
    """
    Visualize the current state of the grid using matplotlib.
    Optionally overlays g/h/f values for each cell.
    """
    plt.clf()
    plt.title('A* Star Search')
    plt.imshow(_get_cmap().get()[map_grid.get()], interpolation='nearest', origin='upper')

    # Overlay g/h/f values if provided
    if cell_values:
        for (r, c), vals in cell_values.items():
            if isinstance(vals, tuple) and len(vals) == 3:
                g_val, h_val, f_val = vals
                text = f"g={g_val}\nh={h_val}\nf={f_val}"
                plt.text(c, r, text, ha='center', va='center', fontsize=6, color='blue')
            else:
                plt.text(c, r, str(vals), ha='center', va='center', fontsize=8, color='red')

    rows, cols = grid_size
    plt.xticks(cp.arange(cols).get(), [str(c) for c in range(cols)])
    plt.yticks(cp.arange(rows).get(), [str(r) for r in range(rows)])
    plt.tick_params(axis='both', which='both', length=0, bottom=True, top=True, left=True, right=True)
    plt.grid(False)
    plt.gca().set_xticks(cp.arange(-.5, cols, 1).get(), minor=True)
    plt.gca().set_yticks(cp.arange(-.5, rows, 1).get(), minor=True)
    plt.gca().grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.axis('image')
    plt.tight_layout()

    if config["show_animation"]:
        plt.pause(config["animation_pause_time"])  # Show animation frame, do NOT close here

    # Only show and close at the very end (final image)
    if config["show_finalimage"] and not config["show_animation"]:
        plt.show(block=True)
        plt.close()

def collect_astar_cell_values(g_val, dest_coords, grid_size):
    """
    Collect g, h, f values for each cell for visualization.
    Returns: dict {(row, col): (g_val, h_val, f_val)}
    """
    cell_values = {}
    for idx, g in g_val.items():
        r, c = ind2sub(idx, grid_size)
        h = abs(r - dest_coords[0]) + abs(c - dest_coords[1])
        f = g + h
        cell_values[(r, c)] = (g, h, f)
    return cell_values

def collect_grassfire_cell_values(wavefront_grid):
    """Collect values for each cell from a wavefront grid (for visualization)."""
    cell_values = {}
    rows, cols = wavefront_grid.shape
    for r in range(rows):
        for c in range(cols):
            cell_values[(r, c)] = wavefront_grid[r, c]
    return cell_values

def visualize_path_on_grid(path, start_coords, dest_coords, visited=None, neighbors=None, color=GRAY, cell_values=None):
    """
    Visualize a given path and optionally visited and neighbor nodes on the grid.
    """
    global map_grid
    # Reset all non-obstacle cells to WHITE
    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            if map_grid[r, c] not in (BLACK, GREEN, YELLOW):
                map_grid[r, c] = WHITE

    # Mark visited nodes
    if visited:
        for node in visited:
            r, c = ind2sub(node, grid_size)
            if map_grid[r, c] not in (BLACK, GREEN, YELLOW):
                map_grid[r, c] = RED

    # Mark neighbor nodes
    if neighbors:
        for node in neighbors:
            r, c = ind2sub(node, grid_size)
            if map_grid[r, c] not in (BLACK, GREEN, YELLOW, RED):
                map_grid[r, c] = BLUE

    # Mark the current path
    for node in path:
        r, c = ind2sub(node, grid_size)
        if (r, c) != tuple(start_coords) and (r, c) != tuple(dest_coords):
            map_grid[r, c] = color

    set_start_and_goal(start_coords, dest_coords)
    if config["show_animation"] or config["show_finalimage"]:
        draw_map(cell_values=cell_values)

# --- Pathfinding Functions ---
def find_route(start_coords, dest_coords):
    """
    Find a single shortest path from start to goal using A*.
    Returns: list of node indices (linear).
    """
    global map_grid, grid_size
    startn = sub2ind(*start_coords, grid_size)
    destn = sub2ind(*dest_coords, grid_size)
    nodeQ = []
    heapq.heappush(nodeQ, (0, startn))
    g = {startn: 0}
    parent = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while nodeQ:
        _, current = heapq.heappop(nodeQ)
        if current == destn:
            route = []
            while current in parent:
                route.append(current)
                current = parent[current]
            route.append(startn)
            route.reverse()
            for node in route[1:-1]:
                r, c = ind2sub(node, grid_size)
                map_grid[r, c] = GRAY
            set_start_and_goal(start_coords, dest_coords)
            if config["show_finalimage"]:
                draw_map()
            return route

        cr, cc = ind2sub(current, grid_size)
        if map_grid[cr, cc] not in (GREEN, YELLOW):
            map_grid[cr, cc] = RED

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
                neighbor = sub2ind(nr, nc, grid_size)
                if map_grid[nr, nc] == BLACK:
                    continue
                tentative_g = g[current] + 1
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    h = abs(nr - dest_coords[0]) + abs(nc - dest_coords[1])
                    f = tentative_g + h
                    parent[neighbor] = current
                    heapq.heappush(nodeQ, (f, neighbor))
                    if map_grid[nr, nc] not in (GREEN, YELLOW):
                        map_grid[nr, nc] = BLUE
        set_start_and_goal(start_coords, dest_coords)
        if config["show_animation"]:
            draw_map()
    return []

def finding_routes(start_coords, dest_coords, max_routes=None, max_depth=None, min_depth=0, max_iterations=500000):
    """
    Find multiple unique shortest paths between start and goal using a modified A* search with iteration limit.
    Returns: (list of shortest paths, cell_values dict)
    """
    global map_grid, grid_size
    startn = sub2ind(*start_coords, grid_size)
    destn = sub2ind(*dest_coords, grid_size)
    open_set = []
    heapq.heappush(open_set, (0, [startn]))
    shortest_routes = []
    shortest_length = None
    seen_paths = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited_nodes = set()
    g_values = {startn: 0}
    cell_values = {}
    iteration_count = 0

    if max_routes is None:
        max_routes = config["max_finding_shortest_routes"]
    if max_depth is None:
        max_depth = config["max_depth_shortest_route"]

    print(f"Finding shortest routes from {start_coords} to {dest_coords}")
    print(f"Max routes: {max_routes}, Min depth: {min_depth}, Max depth: {max_depth}")

    while open_set and len(shortest_routes) < max_routes and iteration_count < max_iterations:
        iteration_count += 1
        f_val, path = heapq.heappop(open_set)
        current = path[-1]
        visited_nodes.add(current)

        if iteration_count % 100000 == 0:
            print(f"Still searching... {iteration_count} iterations so far")

        if current == destn:
            if len(path) < min_depth:
                continue
            if shortest_length is None:
                shortest_length = len(path)
            if len(path) == shortest_length:
                path_tuple = tuple(path)
                if path_tuple not in seen_paths:
                    shortest_routes.append(list(path))
                    seen_paths.add(path_tuple)
                    visualize_path_on_grid(path, start_coords, dest_coords, visited=visited_nodes)
                    print(f"Found shortest route {len(shortest_routes)}/{max_routes}, length: {len(path)}")
            continue

        if shortest_length is not None and len(path) > shortest_length:
            continue
        if len(path) > max_depth:
            continue

        cr, cc = ind2sub(current, grid_size)
        neighbors_this_step = set()
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid_size[0] and 0 <= nc < grid_size[1]:
                neighbor = sub2ind(nr, nc, grid_size)
                if neighbor not in path and map_grid[nr, nc] != BLACK:
                    new_path = path + [neighbor]
                    g_val = len(new_path)
                    g_values[neighbor] = g_val
                    h_val = abs(nr - dest_coords[0]) + abs(nc - dest_coords[1])
                    f_val = g_val + h_val

                    if len(open_set) < config["max_open_set_size"]:
                        heapq.heappush(open_set, (f_val, new_path))
                        neighbors_this_step.add(neighbor)
        if config["show_animation"]:
            cell_values = collect_astar_cell_values(g_values, dest_coords, grid_size)
            visualize_path_on_grid(path, start_coords, dest_coords,
                                   visited=visited_nodes, neighbors=neighbors_this_step, cell_values=cell_values)

    if iteration_count >= max_iterations:
        print(f"Reached iteration limit ({max_iterations}) without finding all routes")

    print(f"Found {len(shortest_routes)} shortest routes after {iteration_count} iterations")
    return shortest_routes, cell_values