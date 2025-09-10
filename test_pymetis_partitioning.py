import networkx as nx
import pymetis

def run_pymetis_partitioning():
    """
    This script demonstrates how to partition a graph with node weights
    using the pymetis library.
    """
    # 1. Define the graph topology and node weights
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    # The METIS algorithm works on the graph directly, unlike Lukes' which
    # required a tree.
    G = nx.Graph(edges)

    # The node weights for nodes 0, 1, 2, 3
    node_weights = [2, 2, 1, 1]

    # 2. Prepare inputs for pymetis
    # pymetis requires the graph as an adjacency list and weights as a list.
    adjacency_list = [list(G.neighbors(i)) for i in sorted(G.nodes())]

    n_parts = 3

    print(f"Partitioning graph into {n_parts} parts...")
    print(f"Nodes: {G.nodes()}")
    print(f"Edges: {G.edges()}")
    print(f"Adjacency List: {adjacency_list}")
    print(f"Node Weights: {node_weights}")

    # 3. Run the partitioning algorithm
    # The function returns the number of cuts and a list where the index is
    # the node and the value is the partition number.
    cuts, partition_for_node = pymetis.part_graph(
        n_parts,
        adjacency=adjacency_list,
        vweights=node_weights
    )

    print(f"\nNumber of edge cuts: {cuts}")
    print(f"Node to partition mapping: {partition_for_node}")

    # 4. Process the output to show the partitions
    partitions = [[] for _ in range(n_parts)]
    for node, part_num in enumerate(partition_for_node):
        partitions[part_num].append(node)

    print(f"\nResulting partitions: {partitions}")

if __name__ == "__main__":
    run_pymetis_partitioning()
