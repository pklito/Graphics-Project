import numpy as np
import cv2 as cv
from util import *
from util import _segmentIntersection
import itertools

class Graph:
    """Key methods of Graph class"""
    def __init__(self):
        #list of points
        self.vertices = dict()
        #dictionary of out edges for each vertex
        self.edges = dict()
        self.max_index = -1
        

    def has_vertex(self, vertex) -> bool:
        return any([(v - vertex).dot(v - vertex) < 0.01 for v in self.vertices.values()])
    
    def get_vertex_index(self, vertex, threshold = 0.01) -> int | None:
        for i, v in self.vertices.pairs():
            if (v - vertex).dot(v - vertex) < threshold * threshold:
                return i
        return None
    
    def add_vertex(self, vertex) -> int:
        vertex = np.asarray(vertex).flatten()
        self.max_index += 1
        self.vertices[self.max_index] = vertex
        self.edges[self.max_index] = set()
        return self.max_index

    def add_edge(self, from_index, to_index) -> None:
        self.edges.setdefault(from_index, set()).add(to_index)
        self.edges.setdefault(to_index, set()).add(from_index)
    
    def swap_vertices(self, v1, v2) -> None:
        # Swap coord values
        self.vertices[v1], self.vertices[v2] = self.vertices[v2], self.vertices[v1]
        # Swap inwards edges (edges of neighbors to v1, and v2)
        for neighbor in self.get_neighbors(v1):
            print(f"neighbor of {v1}: {neighbor}")
            if v2 in self.get_neighbors(neighbor):
                continue
            self.edges[neighbor].remove(v1)
            self.edges[neighbor].add(v2)
        for neighbor in self.get_neighbors(v2):
            if v1 in self.get_neighbors(neighbor):
                continue
            self.edges[neighbor].remove(v2)
            self.edges[neighbor].add(v1)
        # Swap outwards edges of v1 and v2
        self.edges[v1], self.edges[v2] = self.edges[v2], self.edges[v1]

    def remove_vertex(self, vertex_index) -> None:
        # Remove edges pointing to vertex
        for neighbor in self.get_neighbors(vertex_index):
            self.edges[neighbor].remove(vertex_index)

        # Pop new last element
        del self.edges[vertex_index]
        del self.vertices[vertex_index]

    def remove_vertices(self, delete_indices : list) -> None:
        for index in delete_indices:
            self.remove_vertex(index)

    def get_neighbors(self, vertex_index) -> set:
        return self.edges.get(vertex_index, set())
    
    def is_neighbor(self, vertex_index, neighbor_index) -> bool:
        return neighbor_index in self.get_neighbors(vertex_index)
    
    def draw_graph(self, image, edge_color = (0,255,0), vertex_color = (0,255,0), edge_width = 2, vertex_size=5, vertex_numbers = False) -> np.ndarray:
        for i, vertex in self.vertices.items():
            for neighbor in self.get_neighbors(i):
                neighbor_vertex = self.vertices[neighbor]
                cv.line(image, np.array(vertex,dtype=np.int32), np.array(neighbor_vertex,dtype=np.int32), edge_color, edge_width)
            cv.circle(image, np.array(vertex,dtype=np.int32), vertex_size, vertex_color, -1)
            if vertex_numbers:
                cv.putText(image, str(i), tuple(vertex.astype(int) + np.array([0,15])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)
        return image
    
    def __str__(self) -> str:
        return f"Vertices: {[ str(str(i) + str((round(v[0],2), round(v[1],2)))) if v is not None else None for i, v in self.vertices.items()]}, Edges: {self.edges}"
    
    def copy(self) -> 'Graph':
        g = Graph()
        g.max_index = self.max_index
        for i, vertex in self.vertices.items():
            g.vertices[i] = vertex.copy()
        for i, edges in self.edges.items():
            g.edges[i] = edges.copy()
        return g
    
    @property
    def info(self) -> str:
        return f"Vertices: {len(self.vertices)}, Edges: {sum([len(edges) for edges in self.edges.values()])}"
    
    def print_matrix(self) -> None:
        print("X ", end="")
        for i in self.vertices.keys():
            if i % 10 == 0:
                print(i, end = " ")
            else:
                print(i%10, end = " ")
        print("")
        for i in self.vertices.keys():
            print(i, end=" ")
            for j in self.vertices.keys():
                print("1" if i in self.get_neighbors(j) else ".",end=(len(str(j+1)))*" " if (j+1)%10 == 0 else " ")
            print("")
            

def makeGraphFromLines(lines) -> Graph:
    lines = lineMatrixToPairs(lines)
    lines = combineParallelLines(lines)
    g = Graph()
    for i in range(0, len(lines)):
        p1 = g.add_vertex(lines[i][0])
        p2 = g.add_vertex(lines[i][1])
        g.add_edge(p1, p2)
    return g

def mergeOverlappingVertices(const_graph : Graph, threshold = 5, neighbor_limit = None, merge_neighbors = False) -> Graph:
    graph = const_graph.copy()
    """Take a disconnected graph and combine vertices that are close together"""
    for i, vertex in graph.vertices.items():
        if vertex is None:
            continue
        for j, vertex2 in graph.vertices.items():
            # Skip if one of the vertices was removed
            if vertex2 is None:
                continue
            # Skip if the same vertex
            if i == j:
                continue
            # Skip if the vertices are connected
            if not merge_neighbors and (j in graph.get_neighbors(i) or i in graph.get_neighbors(j)):
                continue

            if neighbor_limit is not None and len(graph.get_neighbors(j)) > neighbor_limit:
                continue

            if np.linalg.norm(vertex - vertex2) < threshold:
                for neighbor in graph.get_neighbors(j):
                    graph.add_edge(i, neighbor)
                    graph.add_edge(neighbor, i)

                graph.vertices[j] = None
    
    keys_to_remove = []

    for key, value in graph.vertices.items():
        if type(value) == type(None):
            keys_to_remove.append(key)
    graph.remove_vertices(keys_to_remove)
    return graph

def _sliceEdges(graph : Graph,tree, a,b,c,d, threshold_detect = 5, threshold_splice = 5):
    p1, t, u, ab_len, cd_len = _segmentIntersection(graph.vertices[a], graph.vertices[b], graph.vertices[c], graph.vertices[d], threshold=threshold_detect)
    if p1 is None:
        # No intersection
        return False
    p1_index = c if u < 0.5 else d
    in_ab = t > 0 + threshold_splice/ab_len and t < 1 - threshold_splice/ab_len
    in_cd = u > 0 + threshold_splice/cd_len and u < 1 - threshold_splice/cd_len

    if in_ab:
        p1_index = graph.add_vertex(p1)
        graph.edges[a].remove(b)
        graph.edges[b].remove(a)
        graph.add_edge(p1_index, a)
        graph.add_edge(a, p1_index)
        graph.add_edge(p1_index, b)
        graph.add_edge(b, p1_index)
        

        tree[(a, b)] = (a, p1_index, b)
    else:
        pass
    if in_cd:
        p2_index = graph.add_vertex(p1)
        graph.edges[c].remove(d)
        graph.edges[d].remove(c)
        graph.add_edge(c, p2_index)
        graph.add_edge(p2_index, c)
        graph.add_edge(d, p2_index)
        graph.add_edge(p2_index, d)

        tree[(c, d)] = (c, p2_index, d)
    else:
        pass
    return in_ab or in_cd

def connectIntersectingEdges(const_graph : Graph, threshold_detect = 5, threshold_splice = 0):
    """
    threshold_extend: The threshold for how many pixels edges can be extended by.
    threshold_combine: how close in pixels does the intersection have to be to an end to combine them.
    """
    graph = const_graph.copy()
    replaced_edges = dict()
    # Go over all edges for intersections.
    # All edges means all starting points 'a' and 'c', and all end points 'b' and 'd' where a < b and c < d
    for a, v_a in const_graph.vertices.items():
        for c, v_c in const_graph.vertices.items():
            if c <= a:
                continue
            for b in const_graph.get_neighbors(a):
                for d in const_graph.get_neighbors(c):
                    # if a == d or b == c it's the same edge. if b <= a or d <= c the reverse edge was already found
                    if a == d or b == c or b <= a or d <= c:
                        continue
                    ab_candidates = [(a, b)]
                    cd_candidates = [(c, d)]
                    while any([edge in replaced_edges.keys() for edge in ab_candidates + cd_candidates]):
                        # Replace edges with their replacements (iteratively)
                        # This is done in three stages to maintain edges which don't have replacements?
                        # In more detail, (a,b) -> (a, p1, b) or stays as (a, b)
                        ab_candidates = [replaced_edges.get(edge, edge) for edge in ab_candidates]
                        cd_candidates = [replaced_edges.get(edge, edge) for edge in cd_candidates]
                        # (a, p1, b) -> [(a, p1)] + [(p1, b)]   but (c,d) -> [(c, d)] + [None]
                        ab_candidates = [(edge[0], edge[1]) if len(edge) == 3 else edge for edge in ab_candidates] + [(edge[1], edge[2]) if len(edge) == 3 else None for edge in ab_candidates]
                        cd_candidates = [(edge[0], edge[1]) if len(edge) == 3 else edge for edge in cd_candidates] + [(edge[1], edge[2]) if len(edge) == 3 else None for edge in cd_candidates]
                        # [(a, p1), (p1, b), (c, d), None] -> [(a, p1), (p1, b), (c, d)]
                        ab_candidates = [edge for edge in ab_candidates if edge is not None]
                        cd_candidates = [edge for edge in cd_candidates if edge is not None]
                    
                    for ab, cd in itertools.product(ab_candidates, cd_candidates):
                        # Do all combinations until one is found
                        if _sliceEdges(graph, replaced_edges, *ab, *cd, threshold_detect=threshold_detect, threshold_splice=threshold_splice):
                            break


                    
    return graph

def getFaces(graph : Graph):
    """Return the faces of the graph"""
    faces_list = [] # Preserve vertex order
    faces_set = set() # Prevent repetitions ( there are 7 per face )
    for i in graph.vertices.keys():
        for j in graph.get_neighbors(i):
            if j == i:
                continue
            for k in graph.get_neighbors(j):
                if j == k or i == k:
                    continue
                for l in graph.get_neighbors(k):
                    if l == i or l == j or l == k:
                        continue
                    if i in graph.get_neighbors(l):
                        if frozenset((i, j, k, l)) in faces_set:
                            continue
                        faces_set.add(frozenset((i, j, k, l)))
                        faces_list.append(tuple([graph.vertices[x] for x in [i, j, k, l]]))
    return faces_list

if __name__ == "__main__":
    g = Graph()
    np.random.seed(0)  # For reproducibility
    groups = []
    for _ in range(7):
        amount = np.random.randint(2, 5)
        group = [g.add_vertex((np.random.uniform(0, 600), np.random.uniform(0, 400))) for _ in range(amount)]
        groups.append(group)
        for i in range(len(group)-1):
            g.add_edge(group[i], group[i+1])

    image = np.zeros((400, 600, 3), dtype=np.uint8)
    g.draw_graph(image)
    # Create power group of the elements in groups
    g = connectIntersectingEdges(g, threshold_splice=0, threshold_detect=5)
    g = mergeOverlappingVertices(g, threshold=20)


    g.draw_graph(image, vertex_color=(255,255,255),edge_color=(255,0,0),edge_width=1)
    cv.imshow("Graph", image)
    cv.waitKey(0)
     
    exit()
    g.draw_graph(image)
    cv.imshow('Graph', image)
    cv.waitKey(0)
    g = connectIntersectingEdges(g, threshold_splice=10, threshold_detect=5)
    g.draw_graph(image)
    cv.imshow('Graph2', image)
    cv.waitKey(0)
    cv.destroyAllWindows()