import typing
class Complex(type):
    def __init__(self, decomposition) -> None:
        self.decomposition = decomposition
    def __iter__(self):
        return self.decomposition.__iter__()
    def get_item(self, i : int):
        return self.decomposition[i]
        
class Grid:
    def __init__(self, vertices : typing.List, elements : Complex) -> None:
        self.vertices = vertices
        self.complex = elements
    def refine(self):
        raise NotImplementedError()

class Simplicial_Grid(Grid):
    def __init__(self, vertices: typing.List, elements: Complex) -> None:
        super().__init__(vertices, elements)
    def refine(self):
        #     .
        #   ./_\.
        # ./_\./_\.
        # for each simplex in complex we are adding 4
        pass
    