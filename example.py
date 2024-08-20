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
from defs.utils.Plasticity import cast_to_set, tensor_norm
from matplotlib.animation import FuncAnimation
import time
def right_corner(x):
    return -200
    # if x[0] <= 0.8 and x[0] >= 0.5 :
    #     return -300
    # else : 
    #     return 0
def is_zero(arr):
    flag = True
    for array_el in arr:
        if array_el != 0.0:
            print("nie jest rowne!")
            flag = False
            break
    return flag
class Linear_Operator(Operator):
    def __init__(self, grid, points, enum, n, a, b, dx, dy, grid_shape) -> None:
        self.points = points
        self.enum = enum
        self.n = n
        self.a = a
        self.b = b
        self.dx = dx
        self.dy = dy
        self.shape = grid_shape
        super().__init__(grid)
    def allocate(self) -> None:
        self.A = np.zeros((self.n, self.n))
        self.C = np.zeros((self.n, self.n))
        self.A2 = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n*2, self.n*2))
        self.D = np.zeros((self.n*2))
        self.f1 = np.zeros(self.n)
        self.f2 = np.zeros(self.n)
        self.plasticity_first_part = np.zeros((self.n,self.n))
        self.plasticity_second_part = np.zeros((self.n,self.n))

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
                mixed_integral = integrator.mixed_value_grad_integral()

                self.A[enum.rid][enum2.rid] += self.a * grad + (self.a + self.b) * mix_grad[0][0]
                self.C[enum.rid][enum2.rid] += (self.a + self.b) * mix_grad[0][1]
                self.A2[enum.rid][enum2.rid] += self.a * grad + (self.a + self.b) * mix_grad[1][1]

                self.plasticity_first_part[enum.rid][enum2.rid] += mixed_integral[0]
                self.plasticity_second_part[enum.rid][enum2.rid] += mixed_integral[2]

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
                    neuman.append([trian,p])
            h = lambda x : 100
            h2 = lambda x : right_corner(x)
            if len(neuman) > 1: # NonHomogenus Neuman
                for t, p in neuman:
                    for t2, p2 in neuman:
                        if p.id != p2.id:
                            self.f1[p.rid] += h(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
                            self.f2[p.rid] += h2(p.x)*0.5*np.linalg.norm(np.array(p.x) - np.array(p2.x))
    def asemble_matrices(self) -> None:         # Fills large matrices with values of matrices calculated in fill()
        k = self.n
        self.M = np.zeros((2*k, 2*k))
        self.F = np.zeros(2*k)
        self.plasticity_A = np.zeros((k*2, k*3))
        self.plasticity_B = np.zeros((k*2, k*3))

        self.M[:k,:k] = self.A
        self.M[:k,k:] = self.C
        self.M[k:,:k] = self.C.T
        self.M[k:,k:] = self.A2

        self.F[:k] = self.f1
        self.F[k:] = self.f2

        self.plasticity_A[:k,:k] = self.plasticity_first_part
        self.plasticity_A[k:,2*k:] = self.plasticity_first_part

        self.plasticity_A[:k,2*k:] = self.plasticity_second_part
        self.plasticity_A[k:,k:2*k] = self.plasticity_second_part

        self.plasticity_B[:k,:k] = self.plasticity_first_part
        self.plasticity_B[:k,k:2*k] = self.plasticity_first_part

        self.plasticity_B[k:,:k] = self.plasticity_second_part
        self.plasticity_B[:k,:k] = self.plasticity_second_part
        
        self.Contact_Points = np.diag(self.B)
        self.Contact_Interval_Length = self.Contact_Points * 0.03
        self.Contact_Limits = np.array([ -1000 if x == 0 else x for x in self.Contact_Points * 0.05])

    def solve(self): #-> List[Function]:
        self.asemble_matrices()

        k = self.n
        dt = 0.1        # Time step
        T = 2           # whole elapsed time of system
        viscosity = 1e13   # viscosity parameter
        dispalcement_history = []
        stress_history = []
        start = time.time()

        sigma = np.zeros(3*k)   # stress
        u_0 = np.zeros(2*k)     # displacement

        for i in range(int(T/dt)):
            dispalcement_history.append(u_0)
            stress_history.append(tensor_norm(sigma))

            u = np.linalg.solve(self.M, self.F + self.M @ u_0 - dt * (1/(2*viscosity)) * (2 * self.a * self.plasticity_A + self.b * self.plasticity_B) @ (sigma - cast_to_set(sigma, 430)) - self.plasticity_A @ sigma )
            
            sigma_diference = sigma - cast_to_set(sigma, 430)
            # kolejnosc pochodnych dy, dx jest podejrzana
            grad_ux_dx , grad_ux_dy     = np.gradient(u[:k].reshape(self.shape),self.dx, self.dy)
            grad_ux_dx , grad_ux_dy     = grad_ux_dx.reshape((k)), grad_ux_dy.reshape((k))
            grad_uy_dx , grad_uy_dy     = np.gradient(u[k:].reshape(self.shape), self.dx, self.dy)
            grad_uy_dx , grad_uy_dy     = grad_uy_dx.reshape((k)), grad_uy_dy.reshape((k))           
            grad_u0x_dx, grad_u0x_dy    = np.gradient(u_0[:k].reshape(self.shape),self.dx, self.dy)
            grad_u0x_dx, grad_u0x_dy    = grad_u0x_dx.reshape((k)), grad_u0x_dy.reshape((k))
            grad_u0y_dx, grad_u0y_dy    = np.gradient(u_0[k:].reshape(self.shape), self.dx, self.dy)
            grad_u0y_dx, grad_u0y_dy    = grad_u0y_dx.reshape((k)), grad_u0y_dy.reshape((k))



            sigma[:k] += dt*(-1/(2*viscosity) * (self.b *(sigma_diference[:k] + sigma_diference[k:2*k]) + 2 * self.a * sigma_diference[:k]))
            sigma[k:2*k] += dt*(-1/(2*viscosity) * (self.b *(sigma_diference[:k] + sigma_diference[k:2*k]) + 2 * self.a * sigma_diference[k:2*k]))
            sigma[2*k:] += dt*(-self.a * (1/viscosity) * sigma_diference[2*k:])

            sigma[:k] += (self.b * (grad_ux_dx + grad_uy_dy) + 2 * self.a * grad_ux_dx)
            sigma[k:2*k] += (self.b * (grad_ux_dx + grad_uy_dy) + 2 * self.a * grad_uy_dy)
            sigma[2*k:] += (self.a * (grad_uy_dx + grad_ux_dy))

            sigma[:k] -= self.b * (grad_u0x_dx + grad_u0y_dy) + 2 * self.a * grad_u0x_dx
            sigma[k:2*k] -= self.b * (grad_u0x_dx + grad_u0y_dy) + 2 * self.a * grad_u0y_dy
            sigma[2*k:] -= self.a * (grad_u0y_dx + grad_u0x_dy)
            # sigma = Ce(u) + dt*G(sigma) - Ce(u_0) + sigma
            u_0 = u

        end = time.time()
        print("Elapsed time", end - start)
        return dispalcement_history, stress_history

n = 36
p1=0; k = 0.8; b = 0; e = 1/4
dx = (k - p1) / n
dy = (b - e) / int(n/3)
shape = (int(n/3), n-1)

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
    if p[0] == p1:
        enum.append(Point(p, i, -1, Group.Dirichlet))
    elif p[1] == e:
        enum.append(Point(p, i, r, Group.Neuman))
        r+=1
    elif p[1] == b and p[0]>=0.3 and p[0]<=0.5:
        enum.append(Point(p, i, r, Group.Contact))
        r+=1
    else:
        enum.append(Point(p, i, r, Group.Interior))
        r+=1
    i+=1
    
tri = Delaunay(points)
i = 0
j = 0
opearator = Linear_Operator(tri.simplices, points, enum, r, 80769, 121153, dx, dy, shape)
opearator.construct()
dispacement, stress = opearator.solve()

fig, [ax1, ax2] = plt.subplots(1,2)
def animate(i):
    global points
    sig = stress[i % len(stress)]
    plt.suptitle("T = " + str(i*0.1))
    ax1.clear()
    ax1.set_title("Stress")
    ax1.imshow(sig.reshape(shape), origin='lower',interpolation='bilinear')
    ax2.clear()
    ax2.set_title("Displacement")
    u_t = dispacement[i % len(dispacement)]
    l=0
    pointsz = np.array(points)
    res = np.reshape(u_t, (2,r))
    res = np.transpose(res)
    for i in range(np.shape(pointsz)[0]):
        if pointsz[i][0] == p1:
            pass
        else:
            pointsz[i]+=res[l]
            l+=1
    points = np.transpose(points)
    pointsz = np.transpose(pointsz)
    ax2.triplot(pointsz[0],pointsz[1], tri.simplices, color='blue')
    points = np.transpose(points)

ani = FuncAnimation(
    fig, animate, 20, interval=100)
plt.show()