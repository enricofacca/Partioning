import argparse
import unittest
import numpy as np
import networkx as nx

def build_graph_from_array(array):
    """
    Builds a networkx graph from a 2D array.

    Nodes are created at the coordinates of non-zero entries in the array.
    Edges are created between adjacent (up, down, left, right) nodes.

    Args:
        array (np.ndarray): A 2D numpy array.

    Returns:
        networkx.Graph: The graph of connected non-zero cells.
    """
    G = nx.Graph()

    # Find the coordinates of all non-zero cells
    # np.argwhere returns an array of [row, col] pairs
    active_cells_coords = np.argwhere(array != 0)

    # Use a set for efficient checking of active cells
    active_nodes = {tuple(coords) for coords in active_cells_coords}

    # Add all active cells as nodes
    G.add_nodes_from(active_nodes)

    # Iterate through active cells to add edges
    for r, c in active_nodes:
        # Check neighbor to the right
        neighbor = (r, c + 1)
        if neighbor in active_nodes:
            G.add_edge((r, c), neighbor)

        # Check neighbor below
        neighbor = (r + 1, c)
        if neighbor in active_nodes:
            G.add_edge((r, c), neighbor)

    return G

class TestGraphBuilding(unittest.TestCase):
    def test_user_example(self):
        """
        Tests the graph building function with the specific example
        provided by the user.
        """
        # 1. Define the input array and expected topology
        input_array = np.array([[0, 1, 0],
                                [1, 1, 0]])

        expected_nodes = {(0, 1), (1, 0), (1, 1)}
        expected_edges = {tuple(sorted([(0, 1), (1, 1)])),
                          tuple(sorted([(1, 0), (1, 1)]))}

        # 2. Build the graph
        G = build_graph_from_array(input_array)

        # 3. Assert the properties of the graph
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertEqual(G.number_of_edges(), 2)

        # Check nodes and edges explicitly
        self.assertEqual(set(G.nodes()), expected_nodes)

        # Normalize the actual edges for comparison
        actual_edges = {tuple(sorted(edge)) for edge in G.edges()}
        self.assertEqual(actual_edges, expected_edges)

def main():
    """
    Main function to handle command-line execution and demonstration.
    """
    parser = argparse.ArgumentParser(
        description="Build a graph from a 2D array of 0s and 1s."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the built-in unit tests."
    )
    args = parser.parse_args()

    if args.test:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestGraphBuilding))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # Demonstrate with a sample array
        print("Demonstrating with a sample array (a hollow square)...")
        sample_array = np.zeros((10, 10), dtype=int)
        sample_array[1:9, 1] = 1
        sample_array[1:9, 8] = 1
        sample_array[1, 1:9] = 1
        sample_array[8, 1:9] = 1

        print("Input Array:")
        print(sample_array)

        # Build the graph
        G = build_graph_from_array(sample_array)

        # Print graph properties
        print(f"\nResulting Graph Properties:")
        print(f"  Number of nodes: {G.number_of_nodes()}")
        print(f"  Number of edges: {G.number_of_edges()}")

        # Plot the graph
        try:
            import matplotlib.pyplot as plt
            print("\nPlotting graph...")
            pos = {node: (node[1], -node[0]) for node in G.nodes()}
            nx.draw(G, pos, with_labels=True, node_size=200, font_size=8)
            plt.title("Graph from Array")
            plt.show()
        except ImportError:
            print("\nMatplotlib not found. Skipping plot.")


if __name__ == "__main__":
    main()
