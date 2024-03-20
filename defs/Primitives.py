from enum import Enum
from typing import List, Any
import numpy as np
class Group(Enum):
    Interior = 1
    Dirichlet = 2
    Neuman = 3
    Contact = 4
class Point:
    def __init__(self, x, id, rid, group_id : Group):
        self.x = x
        self.id = id
        self.rid = rid
        self.group_id = group_id

class Map:
    def __init__(self):
        pass
    def shape_function(self, x, y):
        raise NotImplementedError()
    def value(self, x, y):
        return self.shape_function(x,y)
    def gradient_function(self, x, y):
        raise NotImplementedError()
    def gradient(self, x, y):
        return self.gradient_function(self, x, y)
    def area(self):
        raise NotImplementedError()
    def __call__(self, x,y):
        return (self.value(x,y), self.gradient(x,y))

class Integrator:
    def __init__(self, element1 : Map, element2 : Map):
        self.map1 = element1
        self.map2 = element2
    def value_integral(self):
        raise NotImplementedError()
    def gradient_integral(self):
        raise NotImplementedError()

class Element:
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def build(self) -> Integrator:
        raise NotImplementedError()

class Function():
    def __init__(self, dom : int, codom : int, grid, values = np.array([])) -> None:
        self.dom = dom
        self.codom = codom
        self.grid = grid
        self.values = values
    def show(self):
        pass
    def integral(self, func) -> float:
        pass
class Operator():
    def __init__(self, grid) -> None:
        self.grid = grid
        self.allocate()
    def construct(self) -> None:
        for element in self.grid:
            self.fill(element)
    def fill(self, element) -> None:
        raise NotImplementedError()
    def allocate(self) -> None:
        raise NotImplementedError()
    def solve(self) -> Function:
        raise NotImplementedError()


