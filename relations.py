import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import ast

class Relation:
    def __init__(self, nodes, rel_pairs):
        self.nodes = nodes
        self.rel_pairs = rel_pairs

    def is_reflexive(self):
        reflexive = True
        for node in self.nodes:
            if (node, node) not in self.rel_pairs:
                reflexive = False
                break
        return reflexive

    def is_irreflexive(self):
        irreflexive = True
        for node in self.nodes:
            if (node, node) in self.rel_pairs:
                irreflexive = False
                break
        return irreflexive

    def is_symmetric(self):
        symmetric = True
        for (x, y) in self.rel_pairs:
            if (y, x) not in self.rel_pairs:
                symmetric = False
                break
        return symmetric

    def is_transitive(self):
        transitive = True
        for (x, y1) in self.rel_pairs:
            for (y2, z) in self.rel_pairs:
                if y1==y2 and (x, z) not in self.rel_pairs:
                    transitive = False
                    break
        return transitive

    def is_asymmetric(self):
        asymmetric = True
        for (x,y) in self.rel_pairs:
            if((y,x) in self.rel_pairs):
                asymmetric = False
                break
        return asymmetric

    def is_antisymmetric(self):
        antisymmetric = True
        for (x, y) in self.rel_pairs:
            if x != y and (y, x) in self.rel_pairs:
                antisymmetric = False
                break
        return antisymmetric

    def get_reflexive_pairs(self):
        reflexive_pairs = []
        for (x,y) in self.rel_pairs:
            if(x == y):
                reflexive_pairs.append((x,y))
        return reflexive_pairs

    def get_symmetric_pairs(self):
        symmetric_pairs = []
        for (x,y) in self.rel_pairs:
            if(x != y and (y,x) in self.rel_pairs):
                symmetric_pairs.append(((x,y), (y,x)))
        return symmetric_pairs

    def get_transitive_paths(self):
        transitive_paths = []
        for (x,y1) in self.rel_pairs:
            for (y2,z) in self.rel_pairs:
                if(y1==y2 and x != y1 and y2 != z and x!=z and (x,z) in self.rel_pairs):
                    transitive_paths.append(((x,y1), (y2,z), (x,z)))
        return transitive_paths
    
    def get_equivalence_classes(self):
        equivalence_classes = {}
        for (x,y) in self.rel_pairs:
            if(x not in equivalence_classes):
                equivalence_classes[x] = set()
            equivalence_classes[x].add(y)
        return equivalence_classes
    
    def get_relation_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        index_map = {node: i for i, node in enumerate(self.nodes)}
        for (x, y) in self.rel_pairs:
            # Handle nodes that might be in pairs but not explicitly in the nodes list
            if x not in index_map:
                index_map[x] = len(self.nodes)
                self.nodes.append(x)
                # Resize matrix
                new_matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
                new_matrix[:-1, :-1] = matrix
                matrix = new_matrix
            if y not in index_map:
                index_map[y] = len(self.nodes)
                self.nodes.append(y)
                # Resize matrix
                new_matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
                new_matrix[:-1, :-1] = matrix
                matrix = new_matrix

            matrix[index_map[x], index_map[y]] = 1
        return matrix

    def is_equivalence(self):
        return self.is_reflexive() and self.is_symmetric() and self.is_transitive()

    def is_partial_order(self):
        return self.is_reflexive() and self.is_antisymmetric() and self.is_transitive()
    
class HasseDiagram:
    def __init__(self, relation: Relation):
        if(not relation.is_partial_order()):
            raise ValueError("Relation is not a partial order")
        self.relation = relation
        self.nodes = relation.nodes
        self.diagram = self._build_hasse_diagram()
        self.levels = self._compute_levels()
        
    def _build_hasse_diagram(self):
        # Create a copy to avoid modifying the original list of pairs
        filtered_pairs = [p for p in self.relation.rel_pairs if p[0] != p[1]]
        
        # Create a set for quick lookups of pairs to be removed
        to_remove = set()

        for (x, y) in filtered_pairs:
            for (y_prime, z) in filtered_pairs:
                if y == y_prime:
                    # If (x,z) exists and it is not a reflexive pair that was already filtered
                    if (x, z) in self.relation.rel_pairs and x != z:
                        to_remove.add((x,z))
        
        # Return a new list containing only the pairs not in to_remove
        return [p for p in filtered_pairs if p not in to_remove]

    def _compute_levels(self):
        min_elements = [n for n in self.nodes]
        for (x, y) in self.diagram:
            try:
                min_elements.remove(y)
            except ValueError:
                pass
        levels = {}
        q = [(el, 0) for el in min_elements]
        visited = set(min_elements)

        while q:
            curr, level = q.pop(0)
            levels[curr] = max(levels.get(curr, 0), level)
            
            for _, neighbor in filter(lambda edge: edge[0] == curr, self.diagram):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, level + 1))
        return levels
    
    def _find_levels(self, current_level: int, current_elements: list, levels: dict):
        for current_element in current_elements:
            if current_element not in levels:
                levels[current_element] = current_level
            elif levels[current_element] < current_level:
                levels[current_element] = current_level
            next_elements = [y for (x, y) in self.diagram if x == current_element]
            self._find_levels(current_level + 1, next_elements, levels)

def plot_hasse_diagram(relation: Relation):
    try:
        hasse_diagram = HasseDiagram(relation)
        g = nx.DiGraph()
        for node in hasse_diagram.nodes:
            g.add_node(node, level=hasse_diagram.levels.get(node, 0))
        g.add_edges_from(hasse_diagram.diagram)
        
        pos = nx.multipartite_layout(g, subset_key='level', align='horizontal')
        nx.draw(g, pos=pos, with_labels=True)
        plt.show()
    except ValueError as e:
        print(f"Could not plot Hasse diagram: {e}")
    
def plot_graph(relation: Relation):
    g = nx.DiGraph()
    for node in relation.nodes:
        g.add_node(node)
    g.add_edges_from(relation.rel_pairs)
    nx.draw(g, with_labels=True)
    plt.show()

def show_properties(relation: Relation, name: str):
    print('||||||||||||||||||||||||||||||||')
    print('------ ', name, ' ------')
    plot_graph(relation)
    print('- Relation pairs: ', relation.rel_pairs)
    reflexive = relation.is_reflexive()
    print('- Is reflexive: ', reflexive)
    if reflexive:
        print('    - Pairs: ', relation.get_reflexive_pairs())
    symmetric = relation.is_symmetric()
    print('- Is symmetric: ', symmetric)
    if symmetric:
        print('    - Pairs: ', relation.get_symmetric_pairs())
    transitive = relation.is_transitive()
    print('- Is transitive: ', transitive)
    if transitive:
        print('    - Paths: ', relation.get_transitive_paths())
    print('- Is asymmetric: ', relation.is_asymmetric())
    print('- Is antisymmetric: ', relation.is_antisymmetric())
    
    # Ensure all nodes from pairs are in the nodes list for the matrix
    all_nodes = set(relation.nodes)
    for x, y in relation.rel_pairs:
        all_nodes.add(x)
        all_nodes.add(y)
    relation.nodes = sorted(list(all_nodes))

    matrix = relation.get_relation_matrix()
    df = pd.DataFrame(matrix, index=relation.nodes, columns=relation.nodes)
    print('\nRelation matrix:')
    print(df)
    if relation.is_equivalence():
        print('- Equivalence: ', True)
        print('    - Classes: ', relation.get_equivalence_classes())
        partitions = []
        for v in relation.get_equivalence_classes().values():
            if list(v) not in partitions:
                partitions.append(list(v))
        print('    - Partitions: ', [set(p) for p in partitions])
    elif relation.is_partial_order():
        print('- Partial order: ', True)
        try:
            hasse_diagram = HasseDiagram(relation)
            print('- Hasse diagram: ', hasse_diagram.diagram)
            plot_hasse_diagram(relation)
        except ValueError as e:
            print(f"- Could not generate Hasse diagram: {e}")
    print('||||||||||||||||||||||||||||||||')

def main():
    print("Introduce el conjunto de nodos, separados por comas. Ejemplo: 1,2,3,4")
    nodes_input = input("Nodos: ")
    try:
        nodes = [int(n.strip()) for n in nodes_input.split(',')]
    except ValueError:
        print("Error: Asegúrate de que los nodos son números enteros separados por comas.")
        return

    print("\nIntroduce los pares de la relación como una lista de tuplas. Ejemplo: [(1,1), (2,1), (3,2)]")
    pairs_input = input("Pares: ")
    try:
        rel_pairs = ast.literal_eval(pairs_input)
        if not isinstance(rel_pairs, list) or not all(isinstance(p, tuple) for p in rel_pairs):
            raise ValueError
    except (ValueError, SyntaxError):
        print("Error: Formato de pares inválido. Asegúrate de que es una lista de tuplas, como en el ejemplo.")
        return

    # Automatically determine nodes from pairs if nodes list is empty
    if not nodes:
        all_nodes_in_pairs = set()
        for x, y in rel_pairs:
            all_nodes_in_pairs.add(x)
            all_nodes_in_pairs.add(y)
        nodes = sorted(list(all_nodes_in_pairs))
        print(f"Nodos no especificados, inferidos de los pares: {nodes}")


    relation = Relation(nodes, rel_pairs)
    show_properties(relation, "Relación introducida")

if __name__ == "__main__":
    main()
