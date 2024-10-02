import numpy as np
import cv2 as cv
from util import *

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
    
    def draw_graph(self, image) -> np.ndarray:
        for i, vertex in enumerate(self.vertices):
            for neighbor in self.get_neighbors(i):
                neighbor_vertex = self.vertices[neighbor]
                cv.line(image, np.array(vertex,dtype=np.int32), np.array(neighbor_vertex,dtype=np.int32), (0, 255, 0), 2)
            cv.circle(image, np.array(vertex,dtype=np.int32), 5, (0, 0, 255), -1)
        return image
    
    def __str__(self) -> str:
        return f"Vertices: {[(v[0], v[1]) if v is not None else None for v in self.vertices]}, Edges: {self.edges}"
    
    def copy(self) -> 'Graph':
        g = Graph()
        g.vertices = self.vertices.copy()
        g.edges = self.edges.copy()
        return g

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
        print(i, new_to_old_indices[i])
        new_vertices.append(graph.vertices[new_to_old_indices[i]])
        new_edges[i] = {old_to_new_indices[j] for j in graph.get_neighbors(new_to_old_indices[i])}
    graph.vertices = new_vertices
    graph.edges = new_edges
    return graph

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