import unittest
import networkx as nx

class TestLukesPartitioning(unittest.TestCase):
    def test_custom_graph_partitioning(self):
        """
        Tests the Lukes partitioning algorithm on a custom graph to ensure
        it produces the expected partitions based on node weights.
        """
        # 1. Define the graph topology with specific edge weights
        # to ensure a deterministic Minimum Spanning Tree (MST).
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=1)
        G.add_edge(2, 3, weight=1)
        G.add_edge(3, 0, weight=2) # This edge will be excluded from the MST

        # 2. Compute the MST
        # The MST will be the path 0-1-2-3.
        mst = nx.minimum_spanning_tree(G, weight='weight')

        # 3. Define and assign node weights
        # The weights must be Python integers.
        node_weights = {0: 2, 1: 2, 2: 1, 3: 1}
        nx.set_node_attributes(mst, node_weights, name="weight")

        # 4. Run Lukes' partitioning with max_size = 2
        max_size = 2
        partitions_sets = list(nx.algorithms.community.lukes.lukes_partitioning(
            mst, max_size, node_weight="weight"
        ))

        # 5. Convert the result for comparison
        actual_partitions = sorted([sorted(list(p)) for p in partitions_sets])

        # 6. Define the expected result and assert equality
        expected_partitions = sorted([[0], [1], [2, 3]])

        self.assertEqual(actual_partitions, expected_partitions)

if __name__ == '__main__':
    unittest.main()
