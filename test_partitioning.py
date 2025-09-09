# This script demonstrates how to create a 2D grid graph, find its
# minimum spanning tree (MST), and apply Lukes' partitioning algorithm
# using the networkx library. The results are then visualized using matplotlib.

import networkx as nx
import matplotlib.pyplot as plt
import argparse
import numpy as np

def create_grid_graph(nx_dim, ny_dim):
    """
    Generates a 2D grid graph with specified dimensions.

    Args:
        nx_dim (int): The number of nodes in the x-dimension (width).
        ny_dim (int): The number of nodes in the y-dimension (height).

    Returns:
        networkx.Graph: A networkx graph object representing the 2D grid.
    """
    if nx_dim <= 0 or ny_dim <= 0:
        raise ValueError("Grid dimensions must be positive integers.")
    return nx.grid_2d_graph(nx_dim, ny_dim)

def visualize_graphs(original_graph, mst, partitions, k_value, pos):
    """
    Visualizes the original graph, its MST, and the partitioning results.

    Args:
        original_graph (networkx.Graph): The original grid graph.
        mst (networkx.Graph): The Minimum Spanning Tree of the graph.
        partitions (list): A list of sets, where each set is a node partition.
        k_value (int): The 'k' value used for Lukes' partitioning.
        pos (dict): A dictionary of node positions for consistent layout.
    """
    plt.figure(figsize=(20, 6))

    # 1. Plot the Original Grid Graph
    plt.subplot(1, 3, 1)
    nx.draw(
        original_graph,
        pos,
        with_labels=False,
        node_size=60,
        node_color='#3498db',
        edge_color='#bdc3c7'
    )
    plt.title("Original Grid Graph", fontsize=16)

    # 2. Plot the Minimum Spanning Tree (MST)
    plt.subplot(1, 3, 2)
    nx.draw(
        mst,
        pos,
        with_labels=False,
        node_size=60,
        node_color='#2ecc71',
        edge_color='#bdc3c7'
    )
    plt.title("Minimum Spanning Tree (MST)", fontsize=16)

    # 3. Plot the result of Lukes' Partitioning
    ax3 = plt.subplot(1, 3, 3)
    if partitions:
        # Generate a color for each partition
        color_map = plt.get_cmap('viridis')
        colors = [color_map(i / len(partitions)) for i in range(len(partitions))]

        node_colors = {}
        for i, partition in enumerate(partitions):
            for node in partition:
                node_colors[node] = colors[i]

        # Draw nodes with colors corresponding to their partition
        nx.draw_networkx_nodes(
            original_graph,
            pos,
            node_size=60,
            node_color=[node_colors.get(node, 'gray') for node in original_graph.nodes()]
        )
        nx.draw_networkx_edges(original_graph, pos, alpha=0.4, edge_color='#bdc3c7')
        ax3.set_title(f"Lukes' Partitioning (k={k_value})", fontsize=16)
    else:
        # If partitioning fails or returns nothing, display the original graph
        nx.draw(original_graph, pos, with_labels=False, node_size=60, node_color='#e74c3c')
        ax3.set_title(f"Partitioning Failed for k={k_value}", fontsize=16)
        
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the graph creation, analysis, and visualization.
    """
    # --- Setup Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate and analyze a 2D grid graph.")
    parser.add_argument(
        '--nx',
        type=int,
        default=12,
        help='Width of the grid (number of nodes in x-dimension).'
    )
    parser.add_argument(
        '--ny',
        type=int,
        default=8,
        help='Height of the grid (number of nodes in y-dimension).'
    )
    args = parser.parse_args()

    nx_dim = args.nx
    ny_dim = args.ny

    # --- Step 1: Create the Graph ---
    print(f"Creating a {nx_dim}x{ny_dim} grid graph...")
    G = create_grid_graph(nx_dim, ny_dim)

    # Generate positions for the nodes to ensure a consistent grid layout in plots
    pos = {(x, y): (y, -x) for x, y in G.nodes()}

    # --- Step 2: Find the Minimum Spanning Tree ---
    print("Calculating the Minimum Spanning Tree (MST)...")
    mst = nx.minimum_spanning_tree(G)

    # --- Step 3: Run Lukes' Partitioning Algorithm ---
    # This algorithm partitions the graph into k-edge-connected components.
    # A standard grid graph has an edge connectivity of 2, because removing the
    # two edges of a corner node will disconnect it. Therefore, we can only
    # run this algorithm for k=1 or k=2.
    k_value = 2
    print(f"Running Lukes' partitioning algorithm for k={k_value}...")

    # The lukes_partitioning algorithm requires integer weights.
    weight = np.ones(mst.number_of_nodes(), dtype=int)
    weight[1:10] = 2

    # Assign weights to nodes as an attribute. The `node_weight` parameter for
    # `lukes_partitioning` is a string that refers to the node attribute key.
    node_weights = {node: int(weight[i]) for i, node in enumerate(mst.nodes())}
    nx.set_node_attributes(mst, node_weights, name="weight")

    npartition = 2
    
    try:
        # Check if the graph is k-edge-connected before running the algorithm
        if nx.is_k_edge_connected(G, k_value):
            # The second argument to lukes_partitioning is `max_size`, not the
            # number of partitions. We'll calculate a reasonable max_size to
            # aim for roughly `npartition` partitions.
            total_weight = sum(weight)
            max_size = total_weight // npartition

            partitions = list(nx.algorithms.community.lukes.lukes_partitioning(
                mst, max_size, node_weight="weight"
            ))
            print(f"Found {len(partitions)} partition(s).")
        else:
            print(f"Graph is not {k_value}-edge-connected. Partitioning is not possible for this k.")
            partitions = []
            
    except nx.NetworkXError as e:
        print(f"An error occurred during partitioning: {e}")
        partitions = []

    # --- Step 4: Visualize the Results ---
    print("Generating visualizations...")
    visualize_graphs(G, mst, partitions, k_value, pos)


if __name__ == "__main__":
    main()

