import argparse
import numpy as np
import networkx as nx
import pymetis
import matplotlib.pyplot as plt

def partition_array(weight_array, n_parts):
    """
    Partitions a 2D numpy array by treating it as a weighted grid graph.

    Args:
        weight_array (np.ndarray): A 2D numpy array of positive weights.
        n_parts (int): The number of partitions to create.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as the input, where
                    each cell's value is the partition number it belongs to.
    """
    ny_dim, nx_dim = weight_array.shape

    # 1. Create a grid graph corresponding to the array dimensions
    G = nx.grid_2d_graph(nx_dim, ny_dim)
    nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 2. Convert the 2D weight array to a 1D list for pymetis
    # The order must match the sorted node order.
    vweights = [weight_array[node[1], node[0]] for node in nodes]

    # 3. Prepare the adjacency list for pymetis
    adjacency_list = []
    for node in nodes:
        neighbors = G.neighbors(node)
        neighbor_indices = [node_to_idx[n] for n in neighbors]
        adjacency_list.append(neighbor_indices)

    print(f"Partitioning {nx_dim}x{ny_dim} array into {n_parts} parts...")

    # 4. Run the partitioning algorithm
    cuts, partition_for_node = pymetis.part_graph(
        n_parts,
        adjacency=adjacency_list,
        vweights=vweights
    )

    # 5. Create a 2D array representing the partitions
    partition_matrix = np.zeros((ny_dim, nx_dim), dtype=int)
    for i, node in enumerate(nodes):
        x, y = node
        partition_matrix[y, x] = partition_for_node[i]

    print(f"Partitioning complete. Number of edge cuts: {cuts}")

    return partition_matrix


def visualize_array_partitions(partition_matrix, weight_array):
    """
    Visualizes the partitioned array using matplotlib.

    Args:
        partition_matrix (np.ndarray): A 2D array where each cell's value
                                       is its partition number.
        weight_array (np.ndarray): The original 2D array of weights.
    """
    plt.figure(figsize=(10, 10))
    # Use imshow to display the 2D array as a colored grid.
    plt.imshow(partition_matrix, cmap='viridis', interpolation='nearest')

    # Add text labels for the total weight at the barycenter of each partition
    n_parts = np.max(partition_matrix) + 1
    for part_num in range(n_parts):
        # Find the coordinates of the cells in the current partition
        coords = np.argwhere(partition_matrix == part_num)
        if coords.size == 0:
            continue

        # Calculate the barycenter (centroid)
        barycenter_y, barycenter_x = coords.mean(axis=0)

        # Calculate the total weight of the partition
        total_weight = weight_array[coords[:, 0], coords[:, 1]].sum()

        # Add the text label
        plt.text(
            barycenter_x,
            barycenter_y,
            str(total_weight),
            color='white',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold'
        )

    plt.title("Array Partitions")
    plt.colorbar(label="Partition Number")
    plt.show()

def main():
    """
    Main function to handle command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Partition a 2D weighted array."
    )
    parser.add_argument(
        "--nx", type=int, default=50, help="Width of the array."
    )
    parser.add_argument(
        "--ny", type=int, default=50, help="Height of the array."
    )
    parser.add_argument(
        "--n_parts", type=int, default=4, help="Number of partitions."
    )
    args = parser.parse_args()

    # Generate a sample 2D numpy array with varied weights
    # We'll create a high-weight circular region in the center.
    print(f"Generating a {args.ny}x{args.nx} sample array...")
    weights = np.ones((args.ny, args.nx), dtype=int)
    center_y, center_x = args.ny // 2, args.nx // 2
    radius = min(center_y, center_x) // 2

    y, x = np.ogrid[:args.ny, :args.nx]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    weights[dist_from_center <= radius] = 10

    # Run the partitioning
    partition_matrix = partition_array(weights, args.n_parts)

    # Visualize the result
    visualize_array_partitions(partition_matrix, weights)

if __name__ == "__main__":
    main()
