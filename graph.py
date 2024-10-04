import numpy as np
import cv2 as cv
from util import *
from util import _segmentIntersection

class Graph:
    """Key methods of Graph class"""
    def __init__(self):
        #list of points
        self.vertices = []
        #dictionary of out edges for each vertex
        self.edges = {}
        

    def has_vertex(self, vertex) -> bool:
        return any([(v - vertex).dot(v - vertex) < 0.01 for v in self.vertices])
    
    def get_vertex_index(self, vertex, threshold = 0.01) -> int | None:
        for i, v in enumerate(self.vertices):
            if (v - vertex).dot(v - vertex) < threshold * threshold:
                return i
        return None
    
    def add_vertex(self, vertex) -> int:
        vertex = np.asarray(vertex).flatten()
        self.vertices.append(vertex)
        return len(self.vertices) - 1

    def add_edge(self, from_index, to_index) -> None:
        self.edges.setdefault(from_index, set()).add(to_index)
        self.edges.setdefault(to_index, set()).add(from_index)

    def get_neighbors(self, vertex_index) -> set:
        return self.edges.get(vertex_index, set())
    
    def is_neighbor(self, vertex_index, neighbor_index) -> bool:
        return neighbor_index in self.get_neighbors(vertex_index)
    
    def draw_graph(self, image, edge_color = (0,255,0), vertex_color = (0,255,0), edge_width = 2, vertex_size=5) -> np.ndarray:
        for i, vertex in enumerate(self.vertices):
            for neighbor in self.get_neighbors(i):
                neighbor_vertex = self.vertices[neighbor]
                cv.line(image, np.array(vertex,dtype=np.int32), np.array(neighbor_vertex,dtype=np.int32), edge_color, edge_width)
            cv.circle(image, np.array(vertex,dtype=np.int32), vertex_size, vertex_color, -1)
        return image
    
    def __str__(self) -> str:
        return f"Vertices: {[(v[0], v[1]) if v is not None else None for v in self.vertices]}, Edges: {self.edges}"
    
    def copy(self) -> 'Graph':
        g = Graph()
        g.vertices = self.vertices.copy()
        g.edges = self.edges.copy()
        return g
    
    @property
    def info(self) -> str:
        return f"Vertices: {len(self.vertices)}, Edges: {sum([len(edges) for edges in self.edges.values()])}"
    
    def print_matrix(self) -> None:
        print("X ", end="")
        for i in range(len(self.vertices)):
            if i % 10 == 0:
                print(i, end = " ")
            else:
                print(i%10, end = " ")
        print("")
        for i in range(len(self.vertices)):
            print(i, end=" ")
            for j in range(len(self.vertices)):
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

def mergeOverlappingVertices(graph : Graph, threshold = 5):
    graph = graph.copy()
    """Take a disconnected graph and combine vertices that are close together"""
    for i, vertex in enumerate(graph.vertices):
        if graph.vertices[i] is None:
            continue
        for j, vertex2 in enumerate(graph.vertices):
            # Skip if one of the vertices was removed
            if graph.vertices[j] is None:
                continue
            # Skip if the same vertex
            if i == j:
                continue
            # Skip if the vertices are connected
            if j in graph.get_neighbors(i) or i in graph.get_neighbors(j):
                continue

            if np.linalg.norm(vertex - vertex2) < threshold:
                for neighbor in graph.get_neighbors(j):
                    graph.edges[neighbor].remove(j)
                    graph.add_edge(i, neighbor)
                    graph.add_edge(neighbor, i)

                graph.edges[j] = None
                graph.vertices[j] = None
    
    # Remove all the None values from the graph
    new_index = 0
    new_index_2 = -1
    new_to_old_indices = []
    old_to_new_indices = []
    for i in range(len(graph.vertices)):
        if graph.vertices[i] is None:
            new_index += 1
        else:
            new_index_2 += 1
            new_to_old_indices.append(i)
        old_to_new_indices.append(new_index_2)

    new_vertices = []
    new_edges = dict()
    for i in range(len(new_to_old_indices)):
        new_vertices.append(graph.vertices[new_to_old_indices[i]])
        new_edges[i] = {old_to_new_indices[j] for j in graph.get_neighbors(new_to_old_indices[i])}
    graph.vertices = new_vertices
    graph.edges = new_edges
    return graph

def connectIntersectingEdges(graph : Graph, threshold_extend = 0, threshold_combine = 5):
    """
    threshold_extend: The threshold for how many pixels edges can be extended by.
    threshold_combine: how close in pixels does the intersection have to be to an end to combine them.
    """
    graph = graph.copy()
    # Go over all edges for intersections.
    # All edges means all starting points 'a' and 'c', and all end points 'b' and 'd' where a < b and c < d
    for a in range(len(graph.vertices) - 1):
        for c in range(a+1, len(graph.vertices)):
            for b in graph.get_neighbors(a).copy():
                for d in graph.get_neighbors(c).copy():
                    if a == d or b == c or b <= a or d <= c:
                        continue
                    #graph.print_matrix()
                    p1, t, u, ab_len, cd_len = _segmentIntersection(graph.vertices[a], graph.vertices[b], graph.vertices[c], graph.vertices[d], threshold=threshold_extend)
                    if p1 is None:
                        # No intersection
                        continue
                    p1_index = c if u < 0.5 else d
                    in_ab = t > 0 + threshold_combine/ab_len and t < 1 - threshold_combine/ab_len
                    in_cd = u > 0 + threshold_combine/cd_len and u < 1 - threshold_combine/cd_len

                    if in_ab:
                        p1_index = graph.add_vertex(p1)
                        graph.edges[a].remove(b)
                        graph.edges[b].remove(a)
                        graph.add_edge(a, p1_index)
                        graph.add_edge(p1_index, b)
                    else:
                        pass
                    if in_cd:
                        p2_index = graph.add_vertex(p1)
                        graph.edges[c].remove(d)
                        graph.edges[d].remove(c)
                        graph.add_edge(c, p2_index)
                        graph.add_edge(p2_index, d)
                    else:
                        pass
    return graph

def getFaces(graph : Graph):
    """Return the faces of the graph"""
    faces_list = [] # Preserve vertex order
    faces_set = set() # Prevent repetitions ( there are 7 per face )
    for i in range(len(graph.vertices)):
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
    g.add_vertex([1, 2])
    g.add_vertex([2, 3])
    g.add_vertex([3, 4])
    g.add_vertex([4, 5])
    g.add_vertex([5, 6])

    def line_distance(line1, line2):
        """Calculate the distance between two lines"""
        p1, p2 = line1
        p3, p4 = line2

        def point_line_distance(point, line):
            p1, p2 = line
            if np.linalg.norm(p2 - p1) == 0:
                return np.linalg.norm(point - p1)
            return np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)

        return min(point_line_distance(p1, line2), point_line_distance(p2, line2),
                    point_line_distance(p3, line1), point_line_distance(p4, line1))

    # Example usage:
    line1 = np.array([[0, 0], [0, 0]])
    line2 = np.array([[5, 0], [5, 1]])
    distance = line_distance(line1, line2)
    print(f"Distance between lines: {distance}")