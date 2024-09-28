import numpy as np

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

def lineIntersection(a1, a2, b1, b2):
    xdiff = (a1[0] - a2[0], b1[0] - b2[0])
    ydiff = (a1[1] - a2[1], b1[1] - b2[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(a1, a2), det(b1, b2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


if __name__ == "__main__":
    g = Graph()
    g.add_vertex([1, 2])
    g.add_vertex([2, 3])
    g.add_vertex([3, 4])
    g.add_vertex([4, 5])
    g.add_vertex([5, 6])
    print(g.vertices)
    print(g.has_vertex([1, 2]))