import numpy as np
from util import *

class Graph:
    """Key methods of Graph class"""
    def __init__(self, vertices = None, edges = None):
        #list of points
        self.vertices = []
        #dictionary of out edges for each vertex
        self.edges = {}
        

    def has_vertex(self, vertex):
        return any([(v - vertex).dot(v - vertex) < 0.01 for v in self.vertices])
    
    def get_vertex_index(self, vertex, threshold = 0.01):
        for i, v in enumerate(self.vertices):
            if (v - vertex).dot(v - vertex) < threshold:
                return i
        return None
    
    def add_vertex(self, vertex):
        vertex = np.asarray(vertex).flatten()
        self.vertices.append(vertex)

    def add_edge(self, from_index, to_index):
        self.edges.setdefault(from_index, set()).add(to_index)
        self.edges.setdefault(to_index, set()).add(from_index)

    def get_neighbors(self, vertex_index):
        return self.edges.get(vertex_index, set())
    
    def is_neighbor(self, vertex_index, neighbor_index):
        return neighbor_index in self.get_neighbors(vertex_index)

def combineParallelLines(lines):
    
    """Combine lines that are parallel"""
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if np.abs(np.dot(lines[i], lines[j])) > 0.99:
                lines[i] += lines[j]
                lines[j] = None


if __name__ == "__main__":
    g = Graph()
    g.add_vertex([1, 2])
    g.add_vertex([2, 3])
    g.add_vertex([3, 4])
    g.add_vertex([4, 5])
    g.add_vertex([5, 6])
    print(g.vertices)
    print(g.has_vertex([1, 2]))

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