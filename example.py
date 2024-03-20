from math import inf
import scipy
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from defs.Impl import Linear_Element, Linear_Map
from defs.Primitives import Point, Group, Operator, Function
from defs.Solvers import LinearObstacleSolver
import time

class Linear_Operator(Operator):
    def __init__(self, grid, points, enum, n, a, b) -> None:
        self.points = points
        self.enum = enum
        self.n = n
        self.a = a
        self.b = b
        super().__init__(grid)
    def allocate(self) -> None:
        self.A = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.A2 = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n*2, self.n*2))
        self.D = np.zeros((self.n*2))
        self.f1 = np.zeros(self.n)
        self.f2 = np.zeros(self.n)

    def fill(self, triangle) -> None:
        element = Linear_Element([self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]])
        desc = [self.enum[triangle[0]], self.enum[triangle[1]], self.enum[triangle[2]]]
        
        for (node1,enum) in zip(triangle,desc):
            for (node2,enum2) in zip(triangle,desc):
                if enum.group_id == Group.Dirichlet or enum2.group_id == Group.Dirichlet:
                    continue

                integrator = element.build(self.points[node1], self.points[node2])
                grad = integrator.gradient_integral()
                mix_grad = integrator.mixed_gradient_integral()

                self.A[enum.rid][enum2.rid] += self.a * grad + (self.a + self.b) * mix_grad[0][0]
                self.C[enum.rid][enum2.rid] += (self.a + self.b) * mix_grad[0][1]
                self.A2[enum.rid][enum2.rid] += self.a * grad + (self.a + self.b) * mix_grad[1][1]

                if enum.group_id == Group.Contact:
                    self.B[enum.rid+self.n][enum.rid+self.n] = -1
                    self.D[enum.rid+self.n] = 0.05
            tr = []
            tr2 = []
            for i in range(3):
                if self.enum[triangle[i]].group_id != Group.Dirichlet:
                    tr.append(Linear_Map(self.points[triangle[i%3]], self.points[triangle[(i+1)%3]], self.points[triangle[(i+2)%3]]))
                    tr2.append(self.enum[triangle[i]])
            neuman = []
            for trian, p in zip(tr,tr2):
                if p.group_id == Group.Interior:
                    pass
                    # area, err = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[0] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # area2, err2 = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[1] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # self.f1[p.rid] += area
                    # self.f2[p.rid] += area2
                elif p.group_id == Group.Neuman:
                    pass
                    # area, err = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[0] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # area2, err2 = integrate.dblquad(lambda x,y : trian.shape_function(x,y) * function(trian.map_triangle([x,y]))[1] *
                    # trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    # self.f1[p.rid] += area # Homogenus Neuman Condition
                    # self.f2[p.rid] += area2 #Homogenus Neuman Condition
                    neuman.append([trian,p])
            h = lambda x : 0
            h2 = lambda x : -abs(x[0]-0.4) * 10
            if len(neuman) > 1: # NonHomogenus Neuman
                for t, p in neuman:
                    for t2, p2 in neuman:
                        if p.id != p2.id:
                            self.f1[p.rid] += h(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                            self.f2[p.rid] += h2(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
    def solve(self) -> Function:
        k = self.n
        self.M = np.zeros((2*k, 2*k))
        self.F = np.zeros(2*k)
        self.M[:k,:k] = self.A
        self.M[:k,k:] = self.C
        self.M[k:,:k] = self.C.T
        self.M[k:,k:] = self.A2
        self.F[:k] = self.f1
        self.F[k:] = self.f2
        self.Contact_Points = np.diag(self.B)
        self.Contact_Interval_Length = self.Contact_Points * 0.03
        self.Contact_Limits = np.array([ -1000 if x == 0 else x for x in self.Contact_Points * 0.05])

        start = time.time()
        functional = lambda x : 0.5 * x @ self.M @ x  - self.F @ x #+ np.maximum(0, x) @ self.Contact_Interval_Length
        constraint = LinearConstraint(self.B,lb = -inf, ub=self.D)
        counter = 0
        x = minimize(functional, np.zeros(np.shape(self.F)[0]), method='trust-constr', constraints=[constraint])
        x = x.x
        end = time.time()
        print("Elapsed time", end - start)
        functional = lambda x : 0.5 * x @ self.M @ x  - self.F @ x
        print(functional(x))
        return x

n = 27
p1=0; k = 0.8; b = 0; e = 1/4
x = np.linspace(p1,k,n)
y = np.linspace(b,e,int(n/3))
pointsx, pointsy = np.meshgrid(x,y)
vert = []
for i, j in zip(pointsx, pointsy):
    for a,c in zip(i,j):
        vert.append([a,c])

points = np.array(vert)
enum = []
i=0
r=0
for p in points:
    if p[0] == k or p[0] == p1:
        enum.append(Point(p, i, -1, Group.Dirichlet))

    elif p[1] == e:
        enum.append(Point(p, i, r, Group.Neuman))
        r+=1
    elif p[1] == b and p[0]>=0.3 and p[0]<=0.5:
        enum.append(Point(p, i, r, Group.Contact))
        r+=1
    else:
        enum.append(Point(p, i, r ,Group.Interior))
        r+=1
    i+=1
    
tri = Delaunay(points)
i = 0
j = 0
# plt.triplot(np.transpose(points)[0],np.transpose(points)[1], tri.simplices, color='gray')
# plt.ylim(top=0.4)
# plt.plot([p1,k],[e,e],color='blue', linestyle='dashed', label= 'Neuman boundary')
# plt.plot([p1,p1],[b,e],color='red', linestyle='solid', label = 'Dirichlet boundary')
# plt.plot([k,k],[b,e],color='red', linestyle='solid')
# plt.plot([0.2,0.6],[b,b],color='black', label = 'Contact boundary')
# plt.plot([p1,0.2],[b,b],color='blue', linestyle='dashed')
# plt.plot([0.6,0.2],[b,b],color='blue', linestyle='dashed')
# plt.legend()
# plt.show()
opearator = Linear_Operator(tri.simplices, points, enum, r,3,6)
opearator.construct()
res = opearator.solve()
inner = 0
res = np.reshape(res, (2,r))
#points = np.transpose(points)
pointsz = np.array(points)
l=0
res = np.transpose(res)
for i in range(np.shape(pointsz)[0]):
    if pointsz[i][0] == k or pointsz[i][0] == p1:
        pass
    else:
        pointsz[i]+=res[l]
        l+=1
points = np.transpose(points)
pointsz = np.transpose(pointsz)
#plt.triplot(points[0],points[1],tri.simplices, color='red' )
x = np.linspace(0,0.8,60)
#plt.plot(x, -(np.sin(32*x) + 1)/16, color='black')
plt.triplot(pointsz[0],pointsz[1], tri.simplices, color='blue')
plt.plot([0.3,0.5],[-0.05,-0.05], color='black')
plt.ylim(top=0.4)
plt.show()