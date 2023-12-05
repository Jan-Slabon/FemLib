import numpy as np
from scipy import integrate
from  Primitives import Map, Integrator, Element

class Const_map(Map):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def shape_function(self, x, y):
        return 1
    def gradient_function(self, x, y):
        return np.array([0,0])
    def area(self):
        return np.array( [[self.a[0], self.a[1],1],
                [self.b[0], self.b[1], 1],
                [self.c[0], self.c[1], 1]] )

class Linear_map(Map):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def area(self):
        return np.array( [[self.a[0], self.a[1],1],
                [self.b[0], self.b[1], 1],
                [self.c[0], self.c[1], 1]] )

    def map_triangle(self,x):
        matrix =    [[self.b[0] - self.a[0], self.c[0] - self.a[0]],
                    [self.b[1] - self.a[1], self.c[1] - self.a[1]]]
        matrix = np.array(matrix)
        self.a = np.array(self.a)
        x = np.array(x)
        transformed = np.dot(matrix, x) + self.a
        return transformed

    def jacobian(self, x, y):
        matrix =    [[self.b[0] - self.a[0], self.c[0] - self.a[0]],
                    [self.b[1] - self.a[1], self.c[1] - self.a[1]]]
        matrix = np.array(matrix)
        return np.linalg.norm(np.linalg.det(matrix))

    def shape_function(self, x,y):
        x,y = self.map_triangle([x,y])
        matrix =    [[x,y,1],
                     [self.b[0], self.b[1], 1],
                     [self.c[0], self.c[1], 1]]
        matrix2 =    [[self.a[0], self.a[1],1],
                     [self.b[0], self.b[1], 1],
                     [self.c[0], self.c[1], 1]]
        matrix = np.array(matrix)
        matrix2 = np.array(matrix2)
        return np.linalg.det(matrix) / np.linalg.det(matrix2)

    def gradient_function(self, x, y):
        dx = self.b[1] - self.c[1]
        dy = -self.b[0] + self.c[0]
        return np.array([dx, dy] /  np.linalg.det(self.area))



class Linear_Integrator(Integrator):
    def __init__(self, map : Linear_map) -> None:
        super().__init__()
        self.map1 = map
        self.map2 = Linear_map(map.b, map.c, map.a)
        self.map3 = Linear_map(map.c, map.a, map.b)

    def value_integral(self):
        integral1, err1 = integrate.dblquad(lambda x,y : self.map1.shape_function(x,y) * self.map2.shape_function(x,y) *
        self.map1.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
        integral2, err2 = integrate.dblquad(lambda x,y : self.map2.shape_function(x,y) * self.map3.shape_function(x,y) *
        self.map2.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
        integral3, err3 = integrate.dblquad(lambda x,y : self.map1.shape_function(x,y) * self.map3.shape_function(x,y) *
        self.map1.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
        return integral1, integral2, integral3

    def gradient_integral(self):
        df1_df2 = np.kron(self.map1.gradient_function(1,1) , self.map2.gradient_function(1,1)) * np.linalg.norm(np.linalg.det(self.map1.area()))
        df2_df3 = np.kron(self.map2.gradient_function(1,1) , self.map3.gradient_function(1,1)) * np.linalg.norm(np.linalg.det(self.map1.area()))
        df1_df3 = np.kron(self.map1.gradient_function(1,1) , self.map3.gradient_function(1,1)) * np.linalg.norm(np.linalg.det(self.map1.area()))
        return df1_df2, df2_df3, df1_df3

class Const_Integrator(Integrator):
    def __init__(self, map : Const_map) -> None:
        super().__init__()
        self.map = map
    def value_integral(self):
        integral = self.map.area()
        return integral, integral, integral
    def gradient_integral(self):
        raise RuntimeError("Zero order elements are not suited for usage in gradient dependent cases!")




class Linear_Element(Element):
    def __init__(self, integrator: Linear_Integrator) -> None:
        super().__init__(integrator)
class Const_Element(Element):
    def __init__(self, integrator: Const_Integrator) -> None:
        super().__init__(integrator)