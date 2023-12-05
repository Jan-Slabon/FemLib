from math import inf
import scipy
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Triangles import triangle_map, Integaral, point


def show(A):
    for el in A:
        if all([x==0 for x in el]):
            for el2 in el:
                print(el2, end = " ")
            print()
class Elements:
    def __init__(self, triangles, points,enum, n,a,b):
        self.triangles = triangles
        self.points = points
        self.n = n
        self.enum = enum
        self.a = a 
        self.b = b

    def ConstructStiffnesNonHomogenus(self):
        self.A = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.A2 = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n*2, self.n*2))
        self.D = np.zeros((self.n*2))
        for triangle in self.triangles:
            tr = []
            tr2 = []
            for i in range(3):
                if self.enum[triangle[i]].group_id != -1:
                    tr.append(triangle_map(self.points[triangle[i%3]], self.points[triangle[(i+1)%3]], self.points[triangle[(i+2)%3]]))
                    tr2.append(self.enum[triangle[i]])
                tmp = self.enum[triangle[i]]
                if tmp.group_id == -3:
                    self.B[tmp.rid+self.n][tmp.rid+self.n] = -1
                    self.D[tmp.rid+self.n] = 0#(np.sin(32*tmp.x[0]) + 1)/16
            for trian, p in zip(tr, tr2):
                for trian2, p2 in zip(tr,tr2):
                    tmp = Integaral(trian, trian2)
                    tmp2 = self.a * tmp.gradient_integral()
                    self.A[p.rid][p2.rid] += tmp2 + (self.a + self.b) * tmp.partial_integral_1()
                    self.C[p.rid][p2.rid] += (self.a + self.b) * tmp.partial_integral_12()
                    self.A2[p.rid][p2.rid] += tmp2 + (self.a + self.b) * tmp.partial_integral_2()
        
    def Create_Right_NonHomogenus(self, function, boundary):
        self.f1 = np.zeros(self.n)
        self.f2 = np.zeros(self.n)
        for triangle in self.triangles:
            tr = []
            tr2 = []
            for i in range(3):
                if self.enum[triangle[i]].group_id != -1:
                    tr.append(triangle_map(self.points[triangle[i%3]], self.points[triangle[(i+1)%3]], self.points[triangle[(i+2)%3]]))
                    tr2.append(self.enum[triangle[i]])
            neuman = []
            for trian, p in zip(tr,tr2):
                if p.group_id == 0:
                    area, err = integrate.dblquad(lambda x,y : trian.shapeFunction(x,y) * function(trian.map_triangle([x,y]))[0] *
                    trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    area2, err2 = integrate.dblquad(lambda x,y : trian.shapeFunction(x,y) * function(trian.map_triangle([x,y]))[1] *
                    trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    self.f1[p.rid] += area
                    self.f2[p.rid] += area2
                elif p.group_id == -2:
                    area, err = integrate.dblquad(lambda x,y : trian.shapeFunction(x,y) * function(trian.map_triangle([x,y]))[0] *
                    trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    area2, err2 = integrate.dblquad(lambda x,y : trian.shapeFunction(x,y) * function(trian.map_triangle([x,y]))[1] *
                    trian.jacobian(x,y), 0, 1, lambda x : 0, lambda x : 1 - x, epsabs=1.5e-4, epsrel=1.5e-4)
                    self.f1[p.rid] += area # Homogenus Neuman Condition
                    self.f2[p.rid] += area2 #Homogenus Neuman Condition
                    neuman.append([trian,p])
            h = lambda x : 0
            h2 = lambda x : 0
            if len(neuman) > 1: # NonHomogenus Neuman
                for t, p in neuman:
                    for t2, p2 in neuman:
                        if p.id != p2.id:
                            self.f2[p.rid] += h(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                            self.f1[p.rid] += h2(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                         
    
    def AssambleMainMatrix(self):
        k = self.n
        self.M = np.zeros((2*k, 2*k))
        self.F = np.zeros(2*k)
        self.M[:k,:k] = self.A
        self.M[:k,k:] = self.C
        self.M[k:,:k] = self.C.T
        self.M[k:,k:] = self.A2
        self.F[:k] = self.f1
        self.F[k:] = self.f2
    def solve(self):
        functional = lambda x : 0.5 * x @ self.M @ x - self.F @ x
        constraint = LinearConstraint(self.B,lb = -inf, ub=self.D)
        counter = 0
        x = minimize(functional, np.zeros(np.shape(self.F)[0]), method='trust-constr', constraints=[constraint])
        return x.x
        # x = np.linalg.solve(self.M, self.F)
        # return x
            


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
    if p[0] == p1 or p[0] == k:
        enum.append(point(p, i, r,-1))

    elif p[1] == e:
        enum.append(point(p, i,r ,-2))
        r+=1
    elif p[1] == b and p[0]>=0.2 and p[0]<=0.6:
        enum.append(point(p, i, r, -3))
        r+=1
    else:
        enum.append(point(p, i, r ,0))
        r+=1
    i+=1
    
tri = Delaunay(points)
i = 0
j = 0
plt.triplot(np.transpose(points)[0],np.transpose(points)[1], tri.simplices, color='gray')
plt.ylim(top=0.4)
plt.plot([p1,k],[e,e],color='blue', linestyle='dashed', label= 'Neuman boundary')
plt.plot([p1,p1],[b,e],color='red', linestyle='solid', label = 'Dirichlet boundary')
plt.plot([k,k],[b,e],color='red', linestyle='solid')
plt.plot([0.2,0.6],[b,b],color='black', label = 'Contact boundary')
plt.plot([p1,0.2],[b,b],color='blue', linestyle='dashed')
plt.plot([0.6,0.2],[b,b],color='blue', linestyle='dashed')
plt.legend()
plt.show()
elementBuilder = Elements(tri.simplices, points, enum, int(1/3 * n*n) - 2*int(n/3),1,2)#25.74,54.71)
elementBuilder.ConstructStiffnesNonHomogenus()
elementBuilder.Create_Right_NonHomogenus(lambda x : [0,0], lambda x: [0,0])
elementBuilder.AssambleMainMatrix()
res = elementBuilder.solve()
inner = 0
res = np.reshape(res, (2,int(1/3 * n*n) - 2*int(n/3)))
#points = np.transpose(points)
pointsz = np.array(points)
l=0
res = np.transpose(res)
for i in range(np.shape(pointsz)[0]):
    if pointsz[i][0] == p1 or pointsz[i][0] == k:
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
plt.plot([0.2,0.6],[0,0], color='black')
plt.ylim(top=0.4)

plt.show()