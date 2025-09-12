import argparse
import unittest
import numpy as np
import networkx as nx

def build_graph_from_array(array):
    """
    Builds a networkx graph from a 2D array.
    Nodes are created at the coordinates of non-zero entries in the array.
    Edges are created between adjacent (up, down, left, right) nodes.
    """
    G = nx.Graph()
    active_cells_coords = np.argwhere(array != 0)
    active_nodes = {tuple(coords) for coords in active_cells_coords}
    G.add_nodes_from(active_nodes)

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

def transform_graph_to_vertex_representation(cell_graph):
    """
    Transforms a graph of cells into a representation of their corner vertices.

    Args:
        cell_graph (networkx.Graph): A graph where each node is a tuple (r, c)
                                     representing a cell's coordinates.

    Returns:
        A tuple containing:
        - list: A list of dictionaries, where each dictionary represents a
                cell and its four corner vertices.
        - list: A sorted list of unique vertex coordinate tuples.
    """
    cell_definitions = []
    unique_vertices = set()

    for r, c in cell_graph.nodes():
        # Define the four corner vertices for the cell (r, c)
        # in a grid of vertices
        vertices = [
            (r, c),
            (r, c + 1),
            (r + 1, c),
            (r + 1, c + 1)
        ]

        cell_definitions.append({
            'cell': (r, c),
            'vertices': vertices
        })

        unique_vertices.update(vertices)

    return cell_definitions, sorted(list(unique_vertices))

class TestGraphTransformation(unittest.TestCase):
    def test_simple_case(self):
        """
        Tests the graph transformation for a simple 2-node graph.
        """
        # 1. Create a simple input graph with two adjacent cells
        G = nx.Graph()
        G.add_nodes_from([(0, 0), (0, 1)])
        G.add_edge((0, 0), (0, 1))

        # 2. Call the transformation function
        cell_defs, unique_vertices = transform_graph_to_vertex_representation(G)

        # 3. Define expected outputs
        expected_cell_defs = [
            {'cell': (0, 0), 'vertices': [(0, 0), (0, 1), (1, 0), (1, 1)]},
            {'cell': (0, 1), 'vertices': [(0, 1), (0, 2), (1, 1), (1, 2)]}
        ]
        expected_unique_vertices = sorted([(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)])

        # 4. Assert the results
        self.assertEqual(len(cell_defs), 2)
        self.assertEqual(len(unique_vertices), 6)

        # To compare lists of dictionaries, we need a stable sort order.
        # We'll sort them by the 'cell' key.
        sorted_cell_defs = sorted(cell_defs, key=lambda x: x['cell'])
        sorted_expected_cell_defs = sorted(expected_cell_defs, key=lambda x: x['cell'])

        self.assertEqual(sorted_cell_defs, sorted_expected_cell_defs)
        self.assertEqual(unique_vertices, expected_unique_vertices)

def main():
    """
    Main function to handle command-line execution and demonstration.
    """
    parser = argparse.ArgumentParser(
        description="Transform a cell-based graph into a vertex representation."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the built-in unit tests."
    )
    args = parser.parse_args()

    if args.test:
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestGraphTransformation))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # Demonstrate with the user's example from the previous script
        print("Demonstrating with a sample array [[0,1,0],[1,1,0]]...")
        input_array = np.array([[0, 1, 0],
                                [1, 1, 0]])

        print("\nInput Array:")
        print(input_array)

        # 1. Build the initial cell graph
        cell_graph = build_graph_from_array(input_array)
        print(f"\nBuilt cell graph with {cell_graph.number_of_nodes()} nodes and {cell_graph.number_of_edges()} edges.")
        print(f"Nodes: {cell_graph.nodes()}")
        print(f"Edges: {cell_graph.edges()}")

        # 2. Transform the graph
        cell_defs, unique_vertices = transform_graph_to_vertex_representation(cell_graph)

        # 3. Print the results
        print("\n--- Transformation Results ---")
        print("\n1. List of Cells with their Vertices:")
        for definition in cell_defs:
            print(f"  - Cell: {definition['cell']}, Vertices: {definition['vertices']}")

        print("\n2. List of Unique Vertex Coordinates:")
        print(f"  Total unique vertices: {len(unique_vertices)}")
        print(f"  Coordinates: {unique_vertices}")


if __name__ == "__main__":
    main()
