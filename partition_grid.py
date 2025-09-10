import argparse
import unittest
import networkx as nx
import numpy as np
import pymetis

def partition_grid_graph(nx_dim, ny_dim, n_parts):
    """
    Creates a 2D grid graph, assigns weights, and partitions it using pymetis.
    """
    # 1. Create a 2D grid graph
    G = nx.grid_2d_graph(nx_dim, ny_dim)
    # pymetis requires integer nodes from 0 to n-1. We create a mapping.
    nodes = sorted(list(G.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 2. Assign varied node weights
    weights = np.ones(len(nodes), dtype=int)
    for i, node in enumerate(nodes):
        if node[0] < nx_dim / 2 and node[1] < ny_dim / 2:
            weights[i] = 5

    # 3. Prepare inputs for pymetis using integer indices
    adjacency_list = []
    for i, node in enumerate(nodes):
        # Get neighbors, then map them to their integer indices
        neighbors = G.neighbors(node)
        neighbor_indices = [node_to_idx[n] for n in neighbors]
        adjacency_list.append(neighbor_indices)

    print(f"Partitioning {nx_dim}x{ny_dim} grid into {n_parts} parts...")

    # 4. Run the partitioning algorithm
    cuts, partition_for_node = pymetis.part_graph(
        n_parts,
        adjacency=adjacency_list,
        vweights=list(weights)
    )

    # 5. Process the output, mapping indices back to node tuples
    partitions = [[] for _ in range(n_parts)]
    for i, part_num in enumerate(partition_for_node):
        original_node = nodes[i] # Use the index to get the original node tuple
        partitions[part_num].append(original_node)

    print(f"Partitioning complete. Number of edge cuts: {cuts}")
    for i, part in enumerate(partitions):
        print(f"  Partition {i}: {part}")

    return partitions, cuts


def partition_grid_graph_unique_ids(nx_dim, ny_dim, n_parts):
    """
    Creates a 2D grid graph with unique integer nodes and partitions it.
    """
    # 1. Create a graph with integer nodes
    G = nx.Graph()
    nodes = range(nx_dim * ny_dim)
    G.add_nodes_from(nodes)

    # 2. Add edges to form the grid structure
    for i in range(nx_dim):
        for j in range(ny_dim):
            node_id = i * ny_dim + j
            # Connect to the node on the right
            if i < nx_dim - 1:
                right_neighbor_id = (i + 1) * ny_dim + j
                G.add_edge(node_id, right_neighbor_id)
            # Connect to the node below
            if j < ny_dim - 1:
                bottom_neighbor_id = i * ny_dim + (j + 1)
                G.add_edge(node_id, bottom_neighbor_id)

    # 3. Assign node weights
    weights = np.ones(len(nodes), dtype=int)
    for node_id in nodes:
        i = node_id // ny_dim
        j = node_id % ny_dim
        if i < nx_dim / 2 and j < ny_dim / 2:
            weights[node_id] = 5

    # 4. Prepare inputs for pymetis
    adjacency_list = [list(G.neighbors(n)) for n in sorted(G.nodes())]

    print(f"Partitioning {nx_dim}x{ny_dim} grid (unique IDs) into {n_parts} parts...")

    # 5. Run the partitioning algorithm
    cuts, partition_for_node = pymetis.part_graph(
        n_parts,
        adjacency=adjacency_list,
        vweights=list(weights)
    )

    # 6. Process the output
    partitions = [[] for _ in range(n_parts)]
    for node_id, part_num in enumerate(partition_for_node):
        partitions[part_num].append(node_id)

    print(f"Partitioning complete. Number of edge cuts: {cuts}")
    for i, part in enumerate(partitions):
        print(f"  Partition {i}: {part}")

    return partitions, cuts


class TestGridPartitioning(unittest.TestCase):
    def test_3x2_grid(self):
        """
        Tests partitioning on a 3x2 grid to ensure a valid partition is created.
        """
        nx_dim, ny_dim = 3, 2
        n_parts = 2

        partitions, cuts = partition_grid_graph(nx_dim, ny_dim, n_parts)

        self.assertEqual(len(partitions), n_parts)

        original_nodes = set(nx.grid_2d_graph(nx_dim, ny_dim).nodes())
        partitioned_nodes = set(p for part in partitions for p in part)
        self.assertEqual(original_nodes, partitioned_nodes)

        all_nodes_in_partitions = [p for part in partitions for p in part]
        self.assertEqual(len(all_nodes_in_partitions), len(set(all_nodes_in_partitions)))

    def test_3x2_grid_unique_ids(self):
        """
        Tests partitioning on a 3x2 grid with unique integer nodes.
        """
        nx_dim, ny_dim = 3, 2
        n_parts = 2

        partitions, cuts = partition_grid_graph_unique_ids(nx_dim, ny_dim, n_parts)

        # 1. Check if the correct number of partitions was created
        self.assertEqual(len(partitions), n_parts)

        # 2. Check if all nodes are present in the partitions
        original_nodes = set(range(nx_dim * ny_dim))
        partitioned_nodes = set(p for part in partitions for p in part)

        self.assertEqual(original_nodes, partitioned_nodes)

        # 3. Check that each node appears in exactly one partition
        all_nodes_in_partitions = [p for part in partitions for p in part]
        self.assertEqual(len(all_nodes_in_partitions), len(set(all_nodes_in_partitions)))


def main():
    """
    Main function to handle command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Partition a 2D grid graph using METIS."
    )
    parser.add_argument("--nx", type=int, default=10, help="Width of the grid.")
    parser.add_argument("--ny", type=int, default=10, help="Height of the grid.")
    parser.add_argument("--n_parts", type=int, default=4, help="Number of partitions.")
    parser.add_argument("--test", action="store_true", help="Run the built-in unit tests.")
    parser.add_argument(
        "--use-unique-ids",
        action="store_true",
        help="Use unique integer IDs for nodes instead of tuples.",
    )
    args = parser.parse_args()

    if args.test:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestGridPartitioning))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif args.use_unique_ids:
        partition_grid_graph_unique_ids(args.nx, args.ny, args.n_parts)
    else:
        partition_grid_graph(args.nx, args.ny, args.n_parts)

if __name__ == "__main__":
    main()
