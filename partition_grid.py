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
    if len(nodes) < 50:
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
    args = parser.parse_args()

    if args.test:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestGridPartitioning))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        partition_grid_graph(args.nx, args.ny, args.n_parts)

if __name__ == "__main__":
    main()
