import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from config import dataset_config as config
from astar_cupy import grid_size, WHITE, GREEN, YELLOW, GRAY, BLACK

# --- Helper Functions ---
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def plot_route_matplotlib(start, goal, route, obstacles, sample_idx, save_folder):
    grid = np.full(grid_size, WHITE)
    # Mark obstacles
    for r, c in obstacles:
        grid[r, c] = BLACK
    grid[start[0], start[1]] = GREEN
    grid[goal[0], goal[1]] = YELLOW
    for node in route:
        r, c = divmod(node, grid_size[1])
        if grid[r, c] not in (GREEN, YELLOW):
            grid[r, c] = GRAY

    color_map = ListedColormap([
        [1, 1, 1],      # 0 - White - Clear Cell
        [0, 0, 0],      # 1 - Black - Obstacle
        [1, 0, 0],      # 2 - Red - Visited
        [0, 0, 1],      # 3 - Blue - On List
        [0, 1, 0],      # 4 - Green - Start
        [1, 1, 0],      # 5 - Yellow - Goal
        [0.5, 0.5, 0.5] # 6 - Gray - Route Path
    ])
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=color_map, vmin=0, vmax=6, origin='upper', interpolation='nearest')
    plt.xticks(np.arange(grid_size[1]), [str(c) for c in range(grid_size[1])])
    plt.yticks(np.arange(grid_size[0]), [str(r) for r in range(grid_size[0])])
    plt.gca().set_xticks(np.arange(-.5, grid_size[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, grid_size[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.tick_params(axis='both', which='both', length=0, bottom=False, top=False, left=False, right=False)
    plt.title(f"Sample {sample_idx + 1}\nStart {start} to Goal {goal}")
    plt.tight_layout()
    # Save the figure
    ensure_folder_exists(save_folder)
    filename = f"sample_{sample_idx + 1}_start_{start[0]}_{start[1]}_to_goal_{goal[0]}_{goal[1]}.png"
    plt.savefig(os.path.join(save_folder, filename))
    print(f"Saved image: {filename}")
    plt.close()

def visualize_dataset_samples(dataset_file, image_path, num_samples=10):
    # Load dataset
    if dataset_file.endswith(".pkl"):
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
    elif dataset_file.endswith(".json"):
        with open(dataset_file, "r") as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported dataset file format. Use .pkl or .json")

    print(f"Loaded {len(data)} samples from {dataset_file}")
    for idx, sample in enumerate(data[:num_samples]):
        # Extract start, goal, route, obstacles from sample
        context = sample.get("Context", [])
        found_routes = sample.get("Found_Routes", [])
        # Find start and goal cell numbers
        start_cell = next((c for c in context if c.get("State_Status") == 1), None)
        goal_cell = next((c for c in context if c.get("State_Status") == 2), None)
        if not start_cell or not goal_cell:
            print(f"Sample {idx+1}: Start or goal not found, skipping.")
            continue
        start = (int(start_cell["y"]) - 1, int(start_cell["x"]) - 1)
        goal = (int(goal_cell["y"]) - 1, int(goal_cell["x"]) - 1)
        # Route as list of cell numbers
        route = [c["Cell_Number"] - 1 for c in found_routes if isinstance(c, dict) and c.get("Cell_Number", 0) > 0]
        # Obstacles
        obstacles = [(int(c["y"]) - 1, int(c["x"]) - 1) for c in context if c.get("State_Status") == config["status_obstacle"]]
        plot_route_matplotlib(start, goal, route, obstacles, idx, image_path)

# --- Main Visualization ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",  default="val",
                        choices=["train", "val", "test"],
                        help="Which split to visualize (default: val)")
    parser.add_argument("--mode",   default="sol",
                        choices=["sol", "nosol"],
                        help="sol or nosol dataset (default: sol)")
    parser.add_argument("--n",      default=10, type=int,
                        help="Number of samples to visualize (default: 10)")
    parser.add_argument("--format", default="json",
                        choices=["json", "pkl"],
                        help="File format to load (default: json)")
    args, _ = parser.parse_known_args()

    # Resolve dataset file path from config or construct fallback
    ext      = args.format
    cfg_key  = f"dataset_{args.split}_{args.mode}_{ext}_file"
    if cfg_key in config:
        dataset_file = config[cfg_key]
    else:
        dataset_file = os.path.join(
            config["output_dir"],
            f"dataset_{args.split}_{args.mode}.{ext}"
        )

    image_path = os.path.join(config["output_dir"], "route_images",
                              f"{args.split}_{args.mode}")
    print(f"Visualizing {args.n} samples from: {dataset_file}")
    visualize_dataset_samples(dataset_file=dataset_file,
                              image_path=image_path,
                              num_samples=args.n)